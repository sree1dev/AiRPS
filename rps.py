import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from termcolor import colored
import keyboard
import sqlite3
import time
from collections import deque, Counter
import random
import os

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Experience Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Database for Permanent Memory
class GameDatabase:
    def __init__(self, db_name="rps_game.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                state TEXT,
                action INTEGER,
                reward REAL,
                next_state TEXT,
                done INTEGER
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                win_ratio REAL,
                tie_ratio REAL,
                detected_patterns TEXT,
                learning_rate REAL,
                epsilon REAL,
                pattern_reward REAL
            )
        ''')
        self.conn.commit()

    def save_transition(self, game_id, state, action, reward, next_state, done):
        state_str = ','.join(map(str, state))
        next_state_str = ','.join(map(str, next_state))
        self.cursor.execute('''
            INSERT INTO games (game_id, state, action, reward, next_state, done)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (game_id, state_str, action, reward, next_state_str, int(done)))
        self.conn.commit()

    def save_performance(self, game_id, win_ratio, tie_ratio, detected_patterns, learning_rate, epsilon, pattern_reward):
        patterns_str = ';'.join(detected_patterns)
        self.cursor.execute('''
            INSERT INTO performance (game_id, win_ratio, tie_ratio, detected_patterns, learning_rate, epsilon, pattern_reward)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (game_id, win_ratio, tie_ratio, patterns_str, learning_rate, epsilon, pattern_reward))
        self.conn.commit()

    def load_transitions(self, max_games=3):
        self.cursor.execute('SELECT DISTINCT game_id FROM games ORDER BY game_id DESC LIMIT ?', (max_games,))
        game_ids = [row[0] for row in self.cursor.fetchall()]
        transitions = []
        if game_ids:
            placeholders = ','.join('?' for _ in game_ids)
            self.cursor.execute(f'SELECT state, action, reward, next_state, done FROM games WHERE game_id IN ({placeholders})', game_ids)
            for row in self.cursor.fetchall():
                state = np.array([float(x) for x in row[0].split(',')])
                action = int(row[1])
                reward = float(row[2])
                next_state = np.array([float(x) for x in row[3].split(',')])
                done = bool(row[4])
                transitions.append((state, action, reward, next_state, done))
        return transitions

    def load_performance(self):
        self.cursor.execute('SELECT win_ratio, tie_ratio, detected_patterns, learning_rate, epsilon, pattern_reward FROM performance ORDER BY game_id DESC LIMIT 1')
        row = self.cursor.fetchone()
        if row:
            patterns = row[2].split(';') if row[2] else []
            return {
                'win_ratio': row[0],
                'tie_ratio': row[1],
                'detected_patterns': patterns,
                'learning_rate': row[3],
                'epsilon': row[4],
                'pattern_reward': row[5]
            }
        return None

    def clear_old_games(self, max_games=3):
        self.cursor.execute('SELECT DISTINCT game_id FROM games ORDER BY game_id DESC LIMIT ?', (max_games,))
        keep_ids = [row[0] for row in self.cursor.fetchall()]
        if keep_ids:
            placeholders = ','.join('?' for _ in keep_ids)
            self.cursor.execute(f'DELETE FROM games WHERE game_id NOT IN ({placeholders})', keep_ids)
            self.conn.commit()

    def reset_database(self):
        self.conn.close()
        if os.path.exists(self.db_name):
            os.remove(self.db_name)
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self.__init__(self.db_name)

    def close(self):
        self.conn.close()

# Rock Paper Scissors AI
class RPS_AI:
    def __init__(self, state_dim=30, action_dim=3, memory_capacity=3000, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_capacity)
        self.db = GameDatabase()
        
        # Initialize DQN models
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Load previous performance
        perf = self.db.load_performance()
        self.learning_rate = perf['learning_rate'] if perf else 0.001
        self.epsilon = perf['epsilon'] if perf else 0.2
        self.pattern_reward = perf['pattern_reward'] if perf else 3.0
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.gamma = 0.99
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.96
        self.steps = 0
        self.update_target_freq = 10
        self.game_id = 0
        self.prev_tie_ratio = 0.0
        
        # Load recent transitions
        for transition in self.db.load_transitions():
            self.memory.push(*transition)
        
        # Game state
        self.move_map = {'r': 0, 'p': 1, 's': 2}
        self.reverse_map = {0: 'r', 1: 'p', 2: 's'}
        self.move_names = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
        self.history = deque(maxlen=12)
        self.outcomes = deque(maxlen=12)
        self.ai_points = 0
        self.human_points = 0
        self.round_count = 0
        self.max_rounds = 15
        self.pattern_confidence = 0.0

    def get_state(self):
        state = np.zeros(self.state_dim)
        human_moves = [human_move for human_move, _ in self.history]
        
        # Encode pattern frequencies and lengths
        patterns = self.detect_patterns(human_moves)
        for i, (pattern, freq) in enumerate(sorted(patterns.items(), key=lambda x: len(x[0]), reverse=True)[:5]):
            state[i] = freq * 4.0 * self.pattern_confidence
            state[5 + i] = len(pattern) / 4.0  # Normalize pattern length
        
        # Recent moves (weighted)
        for i, (human_move, ai_move) in enumerate(list(self.history)[-5:]):
            state[10 + i] = self.move_map.get(human_move, 0) * (1 + 0.2 * (5 - i))
            state[15 + i] = self.move_map.get(ai_move, 0)
        
        # Outcomes
        for i, outcome in enumerate(list(self.outcomes)[-5:]):
            state[20 + i] = outcome
        
        # Ratios and confidence
        if self.round_count > 0:
            state[25] = self.ai_points / self.round_count
            state[26] = self.human_points / self.round_count
            state[27] = (self.round_count - self.ai_points - self.human_points) / self.round_count
        state[28] = self.pattern_confidence
        state[29] = self.prev_tie_ratio
        return state

    def detect_patterns(self, moves):
        if len(moves) < 2:
            return {}
        patterns = Counter()
        for length in range(2, 7):
            for i in range(len(moves) - length + 1):
                pattern = ''.join(moves[i:i+length])
                weight = 1.0 + 0.7 * (i / len(moves))  # Weight recent patterns
                patterns[pattern] += weight
                for j in range(i + length, len(moves) - length + 1):
                    if moves[j:j+length] == list(pattern):
                        patterns[pattern] += weight
        return {k: v for k, v in patterns.items() if v > 0.5}

    def predict_next_move(self, moves):
        patterns = self.detect_patterns(moves)
        if not patterns:
            # Fallback: counter most frequent move
            move_counts = Counter(moves[-6:])
            if move_counts:
                most_common = move_counts.most_common(1)[0][0]
                return most_common, 0.3
            return None, 0.0
        
        # Find the longest pattern matching recent moves
        candidates = [(p, f) for p, f in patterns.items() if moves[-min(len(p), len(moves)):] == list(p)[-min(len(p), len(moves)):]]
        if not candidates:
            return None, 0.0
        longest_pattern, freq = max(candidates, key=lambda x: (len(x[0]), x[1]))
        pattern_length = len(longest_pattern)
        pattern_index = (len(moves) % pattern_length)
        predicted_move = longest_pattern[pattern_index] if pattern_index < pattern_length else longest_pattern[0]
        confidence = min((freq / sum(patterns.values())) * (1 + 0.5 * (pattern_length / 6)), 1.0)
        return predicted_move, confidence

    def choose_action(self, state, predicted_move, confidence):
        self.steps += 1
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
            print(f"Action: Random ({self.reverse_map[action]})")
            return action
        if predicted_move and confidence >= 0.5:
            if predicted_move == 'r':
                action = self.move_map['p']
            elif predicted_move == 'p':
                action = self.move_map['s']
            elif predicted_move == 's':
                action = self.move_map['r']
            print(f"Action: Pattern-based ({self.reverse_map[action]})")
            return action
        # Fallback: counter most frequent move
        human_moves = [hm for hm, _ in list(self.history)[-6:]]
        if human_moves:
            move_counts = Counter(human_moves)
            most_common = move_counts.most_common(1)[0][0]
            if most_common == 'r':
                action = self.move_map['p']
            elif most_common == 'p':
                action = self.move_map['s']
            elif most_common == 's':
                action = self.move_map['r']
            print(f"Action: Frequent move counter ({self.reverse_map[action]})")
            return action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        action = q_values.argmax().item()
        print(f"Action: DQN ({self.reverse_map[action]})")
        return action

    def determine_winner(self, human_move, ai_move):
        if human_move == ai_move:
            return 0
        if (human_move == 'r' and ai_move == 's') or \
           (human_move == 'p' and ai_move == 'r') or \
           (human_move == 's' and ai_move == 'p'):
            return -1
        return 1

    def evaluate_performance(self):
        if self.round_count < 2:
            return
        win_ratio = self.ai_points / self.round_count
        tie_ratio = (self.round_count - self.ai_points - self.human_points) / self.round_count
        human_moves = [human_move for human_move, _ in self.history]
        patterns = self.detect_patterns(human_moves)
        detected_patterns = list(patterns.keys())
        
        # Update pattern confidence
        predicted_move, self.pattern_confidence = self.predict_next_move(human_moves)
        
        # Save performance
        self.db.save_performance(
            game_id=self.game_id,
            win_ratio=win_ratio,
            tie_ratio=tie_ratio,
            detected_patterns=detected_patterns,
            learning_rate=self.learning_rate,
            epsilon=self.epsilon,
            pattern_reward=self.pattern_reward
        )
        
        # Log patterns and predictions
        if detected_patterns:
            print(f"Detected patterns: {', '.join(detected_patterns)}")
        if predicted_move:
            print(f"Predicted next human move: {predicted_move} (confidence: {self.pattern_confidence:.2f})")
        
        # Adjust if performance is poor
        if tie_ratio > 0.2 or win_ratio < 0.7:
            self.adjust_parameters(tie_ratio, win_ratio, patterns)

    def adjust_parameters(self, tie_ratio, win_ratio, patterns):
        if patterns:
            self.pattern_reward = min(self.pattern_reward + 3.0, 10.0)
            self.epsilon = max(self.epsilon * 0.2, self.epsilon_min)
            self.learning_rate = min(self.learning_rate * (1 + 0.5 * self.pattern_confidence), 0.01)
            print(f"Increased pattern_reward to {self.pattern_reward}, decreased epsilon to {self.epsilon}, learning_rate to {self.learning_rate}")
        else:
            self.pattern_reward = max(self.pattern_reward - 0.5, 0.5)
            self.epsilon = min(self.epsilon * 1.5, 0.5)
            self.learning_rate = max(self.learning_rate * 0.8, 0.0001)
            print(f"Decreased pattern_reward to {self.pattern_reward}, increased epsilon to {self.epsilon}, learning_rate to {self.learning_rate}")
        
        # Update optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Clear memory if performance stagnates
        if tie_ratio > 0.3 and self.round_count > 6:
            print("Clearing memory due to high tie ratio")
            self.memory = ReplayMemory(self.memory.capacity)
            self.db.clear_old_games()
        
        # Update target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.prev_tie_ratio = tie_ratio

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.FloatTensor(batch[4]).to(self.device)
        
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def play_game(self):
        # Reset database if previous game had high tie ratio
        if self.prev_tie_ratio > 0.3:
            print("Resetting database due to high tie ratio in previous game")
            self.db.reset_database()
            self.memory = ReplayMemory(self.memory.capacity)
        
        print("Rock, Paper, Scissors! Enter 'r', 'p', or 's' to play (q to quit).")
        self.round_count = 0
        self.ai_points = 0
        self.human_points = 0
        self.game_id += 1
        
        while self.round_count < self.max_rounds:
            state = self.get_state()
            print(f"\nRound {self.round_count + 1}/{self.max_rounds} | AI: {self.ai_points} | Human: {self.human_points}")
            
            # Wait for human input
            human_move = None
            while human_move not in ['r', 'p', 's']:
                key = keyboard.read_key(suppress=False)
                if key == 'q':
                    print("Game terminated.")
                    self.db.close()
                    return
                if key in ['r', 'p', 's']:
                    human_move = key
                    time.sleep(0.3)
                    keyboard.unhook_all()
                    break
            
            # AI chooses move
            human_moves = [hm for hm, _ in self.history] + [human_move]
            predicted_move, confidence = self.predict_next_move(human_moves)
            ai_action = self.choose_action(state, predicted_move, confidence)
            ai_move = self.reverse_map[ai_action]
            
            # Determine outcome
            outcome = self.determine_winner(human_move, ai_move)
            reward = outcome
            if predicted_move and confidence >= 0.5 and outcome == 1:
                reward += self.pattern_reward
            
            # Update points
            if outcome == 1:
                self.ai_points += 1
            elif outcome == -1:
                self.human_points += 1
            
            # Store move history
            self.history.append((human_move, ai_move))
            self.outcomes.append(outcome)
            
            # Get next state
            next_state = self.get_state()
            done = (self.round_count + 1 == self.max_rounds)
            
            # Store transition
            self.memory.push(state, ai_action, reward, next_state, done)
            self.db.save_transition(self.game_id, state, ai_action, reward, next_state, done)
            
            # Optimize and evaluate
            self.optimize()
            self.evaluate_performance()
            
            # Display moves and result
            print(f"Human: {colored(self.move_names[self.move_map[human_move]], 'blue')}")
            print(f"AI: {colored(self.move_names[ai_action], 'yellow')}")
            if outcome == 1:
                print(colored("AI wins this round!", 'red'))
            elif outcome == -1:
                print(colored("Human wins this round!", 'green'))
            else:
                print("It's a tie!")
            
            self.round_count += 1
        
        # Game over
        print(f"\nGame Over! Final Score -> AI: {self.ai_points} | Human: {self.human_points}")
        if self.ai_points > self.human_points:
            print(colored("AI is the overall winner!", 'red'))
        elif self.human_points > self.ai_points:
            print(colored("Human is the overall winner!", 'green'))
        else:
            print("The game ends in a tie!")
        self.db.close()

if __name__ == "__main__":
    ai = RPS_AI()
    ai.play_game()