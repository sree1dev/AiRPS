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
import pickle

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128]):
        super(DQN, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

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
                done INTEGER,
                rating TEXT,
                patterns TEXT
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
                pattern_reward REAL,
                rating TEXT,
                confidence REAL,
                loss REAL
            )
        ''')
        self.conn.commit()

    def save_transition(self, game_id, state, action, reward, next_state, done, rating, patterns):
        state_str = ','.join(map(str, state))
        next_state_str = ','.join(map(str, next_state))
        patterns_str = ';'.join(patterns) if patterns else ''
        self.cursor.execute('''
            INSERT INTO games (game_id, state, action, reward, next_state, done, rating, patterns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (game_id, state_str, action, reward, next_state_str, int(done), rating, patterns_str))
        self.conn.commit()

    def save_performance(self, game_id, win_ratio, tie_ratio, detected_patterns, learning_rate, epsilon, pattern_reward, rating, confidence, loss):
        patterns_str = ';'.join(detected_patterns)
        self.cursor.execute('''
            INSERT INTO performance (game_id, win_ratio, tie_ratio, detected_patterns, learning_rate, epsilon, pattern_reward, rating, confidence, loss)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (game_id, win_ratio, tie_ratio, patterns_str, learning_rate, epsilon, pattern_reward, rating, confidence, loss))
        self.conn.commit()

    def load_transitions(self, max_games=5, min_rating="neutral"):
        self.cursor.execute('SELECT DISTINCT game_id FROM performance WHERE rating IN ("good", "might_work") ORDER BY game_id DESC LIMIT ?', (max_games,))
        game_ids = [row[0] for row in self.cursor.fetchall()]
        if len(game_ids) < max_games:
            self.cursor.execute('SELECT DISTINCT game_id FROM games WHERE game_id NOT IN (SELECT game_id FROM performance WHERE rating IN ("good", "might_work")) ORDER BY game_id DESC LIMIT ?', (max_games - len(game_ids),))
            game_ids.extend([row[0] for row in self.cursor.fetchall()])
        transitions = []
        if game_ids:
            placeholders = ','.join('?' for _ in game_ids)
            rating_priority = {"good": 4, "might_work": 3, "neutral": 2, "bad": 1}
            min_priority = rating_priority.get(min_rating, 2)
            self.cursor.execute(f'''
                SELECT id, state, action, reward, next_state, done, rating, patterns
                FROM games
                WHERE game_id IN ({placeholders}) AND (
                    rating = "good" OR
                    rating = "might_work" OR
                    rating = "neutral" OR
                    (rating = "bad" AND ? = "bad")
                )
            ''', game_ids + [min_rating])
            for row in self.cursor.fetchall():
                state = np.array([float(x) for x in row[1].split(',')])
                action = int(row[2])
                reward = float(row[3])
                next_state = np.array([float(x) for x in row[4].split(',')])
                done = bool(row[5])
                rating = row[6]
                weight = rating_priority.get(rating, 1)
                transitions.extend([(state, action, reward, next_state, done)] * weight)
        random.shuffle(transitions)
        return transitions[:3000]

    def load_performance(self):
        self.cursor.execute('''
            SELECT win_ratio, tie_ratio, detected_patterns, learning_rate, epsilon, pattern_reward, rating, confidence, loss
            FROM performance
            WHERE rating IN ("good", "might_work", "neutral")
            ORDER BY game_id DESC LIMIT 1
        ''')
        row = self.cursor.fetchone()
        if row:
            patterns = row[2].split(';') if row[2] else []
            return {
                'win_ratio': row[0],
                'tie_ratio': row[1],
                'detected_patterns': patterns,
                'learning_rate': row[3],
                'epsilon': row[4],
                'pattern_reward': row[5],
                'confidence': row[7],
                'loss': row[8]
            }
        return None

    def get_historical_tie_ratio(self, max_games=3):
        self.cursor.execute('SELECT tie_ratio FROM performance ORDER BY game_id DESC LIMIT ?', (max_games,))
        tie_ratios = [row[0] for row in self.cursor.fetchall()]
        return np.mean(tie_ratios) if tie_ratios else 0.2

    def get_historical_performance(self, max_games=10):
        self.cursor.execute('SELECT game_id, win_ratio, tie_ratio, confidence, loss FROM performance ORDER BY game_id DESC LIMIT ?', (max_games,))
        return [{'game_id': row[0], 'win_ratio': row[1], 'tie_ratio': row[2], 'confidence': row[3], 'loss': row[4]} for row in self.cursor.fetchall()]

    def update_transition_rating(self, game_id, transition_id, new_rating):
        self.cursor.execute('UPDATE games SET rating = ? WHERE game_id = ? AND id = ?', (new_rating, game_id, transition_id))
        self.conn.commit()

    def update_performance_rating(self, game_id, new_rating):
        self.cursor.execute('UPDATE performance SET rating = ? WHERE game_id = ?', (new_rating, game_id))
        self.conn.commit()

    def re_evaluate_bad_transitions(self, current_game_id, winning_patterns):
        if not winning_patterns:
            return
        winning_patterns_set = set(winning_patterns)
        self.cursor.execute('SELECT id, game_id, patterns, reward FROM games WHERE rating = "bad" AND game_id != ?', (current_game_id,))
        for row in self.cursor.fetchall():
            transition_id, game_id, patterns_str, reward = row
            if patterns_str:
                patterns = set(patterns_str.split(';'))
                if patterns & winning_patterns_set and reward < 1:
                    self.update_transition_rating(game_id, transition_id, "might_work")

    def re_evaluate_bad_games(self, current_game_id, winning_patterns):
        if not winning_patterns:
            return
        winning_patterns_set = set(winning_patterns)
        self.cursor.execute('SELECT game_id, detected_patterns FROM performance WHERE rating = "bad" AND game_id != ?', (current_game_id,))
        for row in self.cursor.fetchall():
            game_id, patterns_str = row
            if patterns_str:
                patterns = set(patterns_str.split(';'))
                if patterns & winning_patterns_set:
                    self.update_performance_rating(game_id, "might_work")

    def close(self):
        self.conn.commit()
        self.conn.close()

# Meta-Controller for Autonomous Optimization
class MetaController:
    def __init__(self):
        self.q_table = {}  # State -> Action -> Q-value
        self.actions = [
            ('learning_rate', 0.5), ('learning_rate', 2.0),
            ('epsilon', 0.5), ('epsilon', 2.0),
            ('pattern_reward', 0.5), ('pattern_reward', 2.0),
            ('memory_capacity', 0.5), ('memory_capacity', 2.0),
            ('batch_size', 0.5), ('batch_size', 2.0),
            ('confidence_threshold', 0.9), ('confidence_threshold', 1.1),
            ('hidden_dims', [128, 128]), ('hidden_dims', [256, 256]),
            ('action_weights', {'random': 0.3, 'pattern': 0.5, 'dqn': 0.2}),
            ('action_weights', {'random': 0.2, 'pattern': 0.6, 'dqn': 0.2})
        ]
        self.stable_config = None
        self.stable_performance = 0.0
        self.experiment_game_count = 0
        self.experiment_config = None
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.2  # Exploration rate
        self.q_table_file = "meta_q_table.pkl"
        self.load_q_table()

    def discretize_state(self, performance):
        return (
            int(performance['win_ratio'] * 10) / 10,
            int(performance['tie_ratio'] * 10) / 10,
            int(performance['confidence'] * 10) / 10,
            int(performance['loss'] * 10) / 10
        )

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        state_key = str(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {str(a): 0.0 for a in self.actions}
        q_values = self.q_table[state_key]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice([eval(a) if isinstance(a, str) else a for a in best_actions])

    def update_q_table(self, state, action, reward, next_state):
        state_key = str(state)
        action_key = str(action)
        next_state_key = str(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {str(a): 0.0 for a in self.actions}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {str(a): 0.0 for a in self.actions}
        current_q = self.q_table[state_key][action_key]
        next_max_q = max(self.q_table[next_state_key].values())
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action_key] = new_q
        self.save_q_table()

    def save_q_table(self):
        with open(self.q_table_file, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            with open(self.q_table_file, 'rb') as f:
                self.q_table = pickle.load(f)

    def evaluate_performance(self, performance):
        reward = performance['win_ratio'] - 2 * performance['tie_ratio'] + 0.5 * performance['confidence'] - 0.5 * performance['loss']
        return max(-1.0, min(1.0, reward))

    def apply_config(self, config, rps_ai):
        param, value = config
        if param == 'learning_rate':
            rps_ai.learning_rate = max(0.0001, min(0.01, rps_ai.learning_rate * value))
            rps_ai.optimizer = optim.Adam(rps_ai.policy_net.parameters(), lr=rps_ai.learning_rate)
        elif param == 'epsilon':
            rps_ai.epsilon = max(0.01, min(0.5, rps_ai.epsilon * value))
        elif param == 'pattern_reward':
            rps_ai.pattern_reward = max(0.5, min(10.0, rps_ai.pattern_reward * value))
        elif param == 'memory_capacity':
            new_capacity = max(1000, min(10000, int(rps_ai.memory.capacity * value)))
            rps_ai.memory = ReplayMemory(new_capacity)
            for transition in rps_ai.db.load_transitions(min_rating="might_work"):
                rps_ai.memory.push(*transition)
        elif param == 'batch_size':
            rps_ai.batch_size = max(32, min(256, int(rps_ai.batch_size * value)))
        elif param == 'confidence_threshold':
            rps_ai.confidence_threshold = max(0.3, min(0.7, rps_ai.confidence_threshold * value))
        elif param == 'hidden_dims':
            rps_ai.policy_net = DQN(rps_ai.state_dim, rps_ai.action_dim, value).to(rps_ai.device)
            rps_ai.target_net = DQN(rps_ai.state_dim, rps_ai.action_dim, value).to(rps_ai.device)
            rps_ai.target_net.load_state_dict(rps_ai.policy_net.state_dict())
            rps_ai.target_net.eval()
            rps_ai.optimizer = optim.Adam(rps_ai.policy_net.parameters(), lr=rps_ai.learning_rate)
            rps_ai.load_model()  # Try to load compatible weights
        elif param == 'action_weights':
            rps_ai.action_weights = value

# Rock Paper Scissors AI
class RPS_AI:
    def __init__(self, state_dim=30, action_dim=3, memory_capacity=3000, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_capacity)
        self.db = GameDatabase()
        self.meta_controller = MetaController()
        
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.load_model()
        
        perf = self.db.load_performance()
        self.learning_rate = perf['learning_rate'] if perf else 0.001
        self.epsilon = perf['epsilon'] if perf else 0.2
        self.pattern_reward = perf['pattern_reward'] if perf else 3.0
        self.confidence = perf['confidence'] if perf else 0.0
        self.loss = perf['loss'] if perf else 0.0
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.gamma = 0.99
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.96
        self.steps = 0
        self.update_target_freq = 10
        self.game_id = 0
        self.prev_tie_ratio = 0.0
        self.pattern_confidence_history = deque(maxlen=5)
        self.loss_history = deque(maxlen=10)
        self.pattern_counts = Counter()
        self.confidence_threshold = 0.5
        self.action_weights = {'random': 0.2, 'pattern': 0.6, 'dqn': 0.2}
        
        for transition in self.db.load_transitions():
            self.memory.push(*transition)
        
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

    def save_model(self, path="rps_model.pth"):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path="rps_model.pth"):
        if os.path.exists(path):
            try:
                self.policy_net.load_state_dict(torch.load(path))
                self.target_net.load_state_dict(self.policy_net.state_dict())
            except RuntimeError:
                print("Warning: Incompatible model weights; initializing new model")

    def get_state(self):
        state = np.zeros(self.state_dim)
        human_moves = [human_move for human_move, _ in self.history]
        
        patterns = self.detect_patterns(human_moves)
        for i, (pattern, freq) in enumerate(sorted(patterns.items(), key=lambda x: len(x[0]), reverse=True)[:5]):
            state[i] = freq * 4.0 * self.pattern_confidence
            state[5 + i] = len(pattern) / 4.0
        
        for i, (human_move, ai_move) in enumerate(list(self.history)[-5:]):
            state[10 + i] = self.move_map.get(human_move, 0) * (1 + 0.2 * (5 - i))
            state[15 + i] = self.move_map.get(ai_move, 0)
        
        for i, outcome in enumerate(list(self.outcomes)[-5:]):
            state[20 + i] = outcome
        
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
                weight = 1.0 + 0.7 * (i / len(moves))
                patterns[pattern] += weight
                for j in range(i + length, len(moves) - length + 1):
                    if moves[j:j+length] == list(pattern):
                        patterns[pattern] += weight
        return {k: v for k, v in patterns.items() if v > 0.5}

    def predict_next_move(self, moves):
        patterns = self.detect_patterns(moves)
        if not patterns:
            move_counts = Counter(moves[-6:])
            if move_counts:
                most_common = move_counts.most_common(1)[0][0]
                return most_common, 0.3
            return None, 0.0
        
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
        weights = self.action_weights
        choice = random.choices(['random', 'pattern', 'dqn'], weights=[weights['random'], weights['pattern'], weights['dqn']], k=1)[0]
        
        if choice == 'random' or np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
            print(f"Action: Random ({self.reverse_map[action]})")
            return action
        if choice == 'pattern' and predicted_move and confidence >= self.confidence_threshold:
            if predicted_move == 'r':
                action = self.move_map['p']
            elif predicted_move == 'p':
                action = self.move_map['s']
            elif predicted_move == 's':
                action = self.move_map['r']
            print(f"Action: Pattern-based ({self.reverse_map[action]})")
            return action
        human_moves = [hm for hm, _ in list(self.history)[-6:]]
        if human_moves and choice != 'dqn':
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
        
        for pattern in detected_patterns:
            self.pattern_counts[pattern] += 1
        
        predicted_move, self.pattern_confidence = self.predict_next_move(human_moves)
        self.pattern_confidence_history.append(self.pattern_confidence)
        self.confidence = self.pattern_confidence
        self.loss = np.mean(self.loss_history) if self.loss_history else 0.0
        
        rating = "good" if win_ratio >= 0.7 and tie_ratio <= 0.2 else "neutral" if tie_ratio <= 0.3 else "bad"
        self.db.save_performance(
            game_id=self.game_id,
            win_ratio=win_ratio,
            tie_ratio=tie_ratio,
            detected_patterns=detected_patterns,
            learning_rate=self.learning_rate,
            epsilon=self.epsilon,
            pattern_reward=self.pattern_reward,
            rating=rating,
            confidence=self.confidence,
            loss=self.loss
        )
        
        if rating == "good" and detected_patterns:
            self.db.re_evaluate_bad_transitions(self.game_id, detected_patterns)
            self.db.re_evaluate_bad_games(self.game_id, detected_patterns)
        
        if detected_patterns:
            print(f"Detected patterns: {', '.join(detected_patterns)}")
        if predicted_move:
            print(f"Predicted next human move: {predicted_move} (confidence: {self.pattern_confidence:.2f})")
        
        performance = {'win_ratio': win_ratio, 'tie_ratio': tie_ratio, 'confidence': self.confidence, 'loss': self.loss}
        self.meta_controller.experiment_game_count += 1
        
        historical_tie_ratio = self.db.get_historical_tie_ratio()
        tie_threshold = max(0.15, historical_tie_ratio * 1.1)
        win_threshold = 0.7 - 0.05 * len(detected_patterns) if detected_patterns else 0.7
        confidence_threshold = 0.5 if len(detected_patterns) <= 2 else 0.4
        
        pattern_stagnation = any(self.pattern_counts[pattern] > 5 and self.ai_points / self.round_count < 0.5 for pattern in detected_patterns)
        high_loss = len(self.loss_history) == 10 and np.mean(self.loss_history) > 1.0
        
        stagnation_score = (
            (1 if tie_ratio > tie_threshold else 0) +
            (1 if win_ratio < win_threshold else 0) +
            (1 if len(self.pattern_confidence_history) == 5 and max(self.pattern_confidence_history) < confidence_threshold else 0) +
            (1 if pattern_stagnation else 0) +
            (1 if high_loss else 0)
        )
        
        state = self.meta_controller.discretize_state(performance)
        reward = self.meta_controller.evaluate_performance(performance)
        
        if self.meta_controller.experiment_config:
            next_state = state
            self.meta_controller.update_q_table(self.meta_controller.experiment_state, self.meta_controller.experiment_config, reward, next_state)
        
        if stagnation_score >= 2 or self.meta_controller.experiment_game_count >= 2:
            print(f"Meta-Controller: Evaluating performance (stagnation_score: {stagnation_score}/5, reward: {reward:.2f})")
            if reward > self.meta_controller.stable_performance:
                self.meta_controller.stable_config = {
                    'learning_rate': self.learning_rate,
                    'epsilon': self.epsilon,
                    'pattern_reward': self.pattern_reward,
                    'memory_capacity': self.memory.capacity,
                    'batch_size': self.batch_size,
                    'confidence_threshold': self.confidence_threshold,
                    'hidden_dims': [layer.out_features for layer in self.policy_net.net if isinstance(layer, nn.Linear)][:-1],
                    'action_weights': self.action_weights
                }
                self.meta_controller.stable_performance = reward
                print(f"Meta-Controller: Updated stable configuration (reward: {reward:.2f})")
            
            if stagnation_score >= 2 or reward < 0.5:
                action = self.meta_controller.get_action(state)
                self.meta_controller.apply_config(action, self)
                self.meta_controller.experiment_config = action
                self.meta_controller.experiment_state = state
                self.meta_controller.experiment_game_count = 0
                print(f"Meta-Controller: Applying new configuration: {action}")
            elif reward < self.meta_controller.stable_performance - 0.1 and self.meta_controller.stable_config:
                for key, value in self.meta_controller.stable_config.items():
                    if key == 'learning_rate':
                        self.learning_rate = value
                        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
                    elif key == 'memory_capacity':
                        self.memory = ReplayMemory(value)
                        for transition in self.db.load_transitions(min_rating="might_work"):
                            self.memory.push(*transition)
                    elif key == 'hidden_dims':
                        self.policy_net = DQN(self.state_dim, self.action_dim, value).to(self.device)
                        self.target_net = DQN(self.state_dim, self.action_dim, value).to(self.device)
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                        self.target_net.eval()
                        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
                    else:
                        setattr(self, key, value)
                self.meta_controller.experiment_config = None
                self.meta_controller.experiment_game_count = 0
                print(f"Meta-Controller: Reverted to stable configuration (reward: {reward:.2f})")
            else:
                action = self.meta_controller.get_action(state)
                self.meta_controller.apply_config(action, self)
                self.meta_controller.experiment_config = action
                self.meta_controller.experiment_state = state
                self.meta_controller.experiment_game_count = 0
                print(f"Meta-Controller: Experimenting with new configuration: {action}")

    def adjust_parameters(self, tie_ratio, win_ratio, patterns, is_stagnant, stagnation_score):
        if is_stagnant and not self.meta_controller.experiment_config:
            self.epsilon = min(self.epsilon * (1.5 + 0.1 * stagnation_score), 0.5)
            self.learning_rate = max(self.learning_rate * (0.8 - 0.05 * stagnation_score), 0.0001)
            self.pattern_reward = max(self.pattern_reward - (0.5 + 0.1 * stagnation_score), 0.5)
            print(f"Stagnation adjustments: Increased epsilon to {self.epsilon}, decreased learning_rate to {self.learning_rate}, pattern_reward to {self.pattern_reward}")
            self.memory = ReplayMemory(self.memory.capacity)
            min_rating = "might_work" if stagnation_score < 4 else "good"
            for transition in self.db.load_transitions(min_rating=min_rating):
                self.memory.push(*transition)
        elif not self.meta_controller.experiment_config:
            if patterns:
                self.pattern_reward = min(self.pattern_reward + 3.0, 10.0)
                self.epsilon = max(self.epsilon * 0.2, self.epsilon_min)
                self.learning_rate = min(self.learning_rate * (1 + 0.5 * self.pattern_confidence), 0.01)
                print(f"Patterns detected: Increased pattern_reward to {self.pattern_reward}, decreased epsilon to {self.epsilon}, learning_rate to {self.learning_rate}")
            else:
                self.pattern_reward = max(self.pattern_reward - 0.5, 0.5)
                self.epsilon = min(self.epsilon * 1.5, 0.5)
                self.learning_rate = max(self.learning_rate * 0.8, 0.0001)
                print(f"No patterns: Decreased pattern_reward to {self.pattern_reward}, increased epsilon to {self.epsilon}, learning_rate to {self.learning_rate}")
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
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
        self.loss_history.append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def play_game(self):
        print("Rock, Paper, Scissors! Enter 'r', 'p', or 's' to play (q to quit).")
        self.round_count = 0
        self.ai_points = 0
        self.human_points = 0
        self.game_id += 1
        transition_ids = []
        
        while self.round_count < self.max_rounds:
            state = self.get_state()
            print(f"\nRound {self.round_count + 1}/{self.max_rounds} | AI: {self.ai_points} | Human: {self.human_points}")
            
            human_move = None
            while human_move not in ['r', 'p', 's']:
                key = keyboard.read_key(suppress=False)
                if key == 'q':
                    print("Game terminated.")
                    self.save_model()
                    self.db.close()
                    return
                if key in ['r', 'p', 's']:
                    human_move = key
                    time.sleep(0.3)
                    keyboard.unhook_all()
                    break
            
            human_moves = [hm for hm, _ in self.history] + [human_move]
            predicted_move, confidence = self.predict_next_move(human_moves)
            ai_action = self.choose_action(state, predicted_move, confidence)
            ai_move = self.reverse_map[ai_action]
            
            outcome = self.determine_winner(human_move, ai_move)
            reward = outcome
            if predicted_move and confidence >= self.confidence_threshold and outcome == 1:
                reward += self.pattern_reward
            
            patterns = self.detect_patterns(human_moves)
            rating = "good" if reward >= 1 else "neutral" if reward == 0 else "bad"
            
            if outcome == 1:
                self.ai_points += 1
            elif outcome == -1:
                self.human_points += 1
            
            self.history.append((human_move, ai_move))
            self.outcomes.append(outcome)
            
            next_state = self.get_state()
            done = (self.round_count + 1 == self.max_rounds)
            
            self.memory.push(state, ai_action, reward, next_state, done)
            self.db.save_transition(self.game_id, state, ai_action, reward, next_state, done, rating, list(patterns.keys()))
            self.cursor = self.db.conn.cursor()
            self.cursor.execute('SELECT last_insert_rowid()')
            transition_id = self.cursor.fetchone()[0]
            transition_ids.append(transition_id)
            
            self.optimize()
            self.evaluate_performance()
            
            print(f"Human: {colored(self.move_names[self.move_map[human_move]], 'blue')}")
            print(f"AI: {colored(self.move_names[ai_action], 'yellow')}")
            if outcome == 1:
                print(colored("AI wins this round!", 'red'))
            elif outcome == -1:
                print(colored("Human wins this round!", 'green'))
            else:
                print("It's a tie!")
            
            self.round_count += 1
        
        if self.round_count >= 2:
            win_ratio = self.ai_points / self.round_count
            tie_ratio = (self.round_count - self.ai_points - self.human_points) / self.round_count
            if win_ratio >= 0.7 and tie_ratio <= 0.2:
                for tid in transition_ids:
                    self.db.update_transition_rating(self.game_id, tid, "good")
            elif tie_ratio > 0.3:
                for tid in transition_ids:
                    self.db.update_transition_rating(self.game_id, tid, "bad")
        
        print(f"\nGame Over! Final Score -> AI: {self.ai_points} | Human: {self.human_points}")
        if self.ai_points > self.human_points:
            print(colored("AI is the overall winner!", 'red'))
        elif self.human_points > self.ai_points:
            print(colored("Human is the overall winner!", 'green'))
        else:
            print("The game ends in a tie!")
        self.save_model()
        self.db.close()

if __name__ == "__main__":
    ai = RPS_AI()
    ai.play_game()