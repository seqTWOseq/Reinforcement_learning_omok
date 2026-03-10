import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tkinter as tk
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
from collections import deque

from khy_model import OmokCNN

# ==========================================
# 1. 오목 강화학습 환경 (GUI 포함)
# ==========================================
class OmokEnvGUI(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.board_size = 15
        self.render_mode = render_mode
        
        # 행동 공간 (0~224) 및 상태 공간 (15x15 배열, 0:빈칸, 1:흑, 2:백)
        self.action_space = spaces.Discrete(self.board_size ** 2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.board_size, self.board_size), dtype=np.int8)
        
        self.board = None
        self.current_player = 1 # 1: 흑, 2: 백
        
        # GUI 설정
        self.window = None
        self.cell_size, self.margin = 40, 30

    def reset(self, seed=None, options=None):
        """보드를 초기화하고 흑돌 턴으로 시작합니다."""
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        return self.board.copy(), {"current_player": self.current_player}

    def step(self, action):
        """에이전트의 행동을 적용하고 상태, 보상, 종료 여부를 반환합니다."""
        r, c = action // self.board_size, action % self.board_size
        
        # 1. 반칙 (이미 돌이 있는 곳) 처리
        if self.board[r, c] != 0:
            return self.board.copy(), -10.0, True, False, {"reason": "invalid_move", "winner": 3 - self.current_player}

        # 2. 착수 및 승리/무승부 판정
        self.board[r, c] = self.current_player
        if self._check_win(r, c, self.current_player):
            return self.board.copy(), 1.0, True, False, {"reason": "win", "winner": self.current_player}
        if not np.any(self.board == 0):
            return self.board.copy(), 0.0, True, False, {"reason": "draw", "winner": 0}

        # 3. 턴 교체
        self.current_player = 3 - self.current_player
        return self.board.copy(), 0.0, False, False, {"current_player": self.current_player}

    def _check_win(self, row, col, player):
        """4방향 탐색으로 5목 완성 여부를 논리적으로 확인합니다."""
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        for dr, dc in directions:
            count = 1
            for step in (1, -1): # 정방향(1), 역방향(-1) 탐색
                r, c = row + dr * step, col + dc * step
                while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                    r += dr * step; c += dc * step
            if count >= 5: return True
        return False

    def render(self):
        """Tkinter 창을 새로고침하여 현재 보드를 시각화합니다."""
        if self.render_mode != "human": return
        
        if self.window is None:
            self.window = tk.Tk()
            self.window.title("AI 오목 대전 시뮬레이터")
            size = (self.board_size - 1) * self.cell_size + self.margin * 2
            self.canvas = tk.Canvas(self.window, width=size, height=size, bg="#DCB35C")
            self.canvas.pack()

        self.canvas.delete("all")
        
        # 격자선 그리기
        for i in range(self.board_size):
            start, end = self.margin + i * self.cell_size, self.margin + (self.board_size - 1) * self.cell_size
            self.canvas.create_line(self.margin, start, end, start)
            self.canvas.create_line(start, self.margin, start, end)

        # 돌 그리기
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] != 0:
                    x, y = self.margin + c * self.cell_size, self.margin + r * self.cell_size
                    rad = self.cell_size // 2 - 2
                    color = "black" if self.board[r, c] == 1 else "white"
                    self.canvas.create_oval(x - rad, y - rad, x + rad, y + rad, fill=color, outline="black")

        # 화면 비동기 업데이트 (mainloop 대체)
        self.window.update_idletasks()
        self.window.update()

    def close(self):
        if self.window: self.window.destroy(); self.window = None

# ==========================================
# 2. 에이전트 클래스 (여기서부터 알고리즘 작성 필요)
# ==========================================
class HumanAgent:
    def __init__(self, env, name="Human(👤)"):
        self.name = name
        self.env = env
        self.clicked_action = None
        self.current_state = None

    def select_action(self, state):
        self.clicked_action = None
        self.current_state = state
        
        # 마우스 왼쪽 클릭 이벤트 활성화
        self.env.canvas.bind("<Button-1>", self._click_handler)
        
        # 클릭될 때까지 무한 대기 (화면은 멈추지 않도록 update 호출)
        while self.clicked_action is None:
            self.env.window.update()
            time.sleep(0.05) # CPU 과부하 방지
            
        # 행동 결정 완료 시 이벤트 해제 (AI 턴에 클릭 방지)
        self.env.canvas.unbind("<Button-1>")
        
        return self.clicked_action

    def _click_handler(self, event):
        # 클릭한 픽셀 위치를 오목판 논리적 좌표(행, 열)로 변환
        c = round((event.x - self.env.margin) / self.env.cell_size)
        r = round((event.y - self.env.margin) / self.env.cell_size)
        
        # 보드 범위 내인지 확인
        if 0 <= r < self.env.board_size and 0 <= c < self.env.board_size:
            action = r * self.env.board_size + c
            
            # 유효한 빈칸(0)을 클릭했을 때만 값 업데이트 (반칙 클릭 무시)
            if self.current_state.flatten()[action] == 0:
                self.clicked_action = action

class HeuristicAgent:
    def __init__(self, name="Heuristic_AI"):
        self.name = name

    def select_action(self, state):
        board_size = state.shape[0]
        valid_moves = np.where(state.flatten() == 0)[0]
        if len(valid_moves) == 0: return 0

        # 보드가 비어있다면 중앙(7, 7)에 착수
        if np.sum(state != 0) == 0:
            return (board_size // 2) * board_size + (board_size // 2)

        best_score = -float('inf')
        best_actions = []

        # 연산 시간을 줄이기 위해 기존 돌 주변 반경 2칸 이내의 빈칸만 탐색 후보로 선정
        candidates = self._get_candidate_moves(state, board_size)
        if not candidates:
            candidates = valid_moves

        for action in candidates:
            r, c = action // board_size, action % board_size
            
            # 내가 착수했을 때의 공격 점수 (Player 1 기준)
            offense_score = self._evaluate_position(state, r, c, player=1)
            # 상대가 착수했을 때의 방어 점수 (Player 2 기준)
            defense_score = self._evaluate_position(state, r, c, player=2)
            
            # 방어에 약간의 가중치를 더 주어 안정적으로 플레이 (1.1배)
            total_score = offense_score + (defense_score * 1.1)

            if total_score > best_score:
                best_score = total_score
                best_actions = [action]
            elif total_score == best_score:
                best_actions.append(action) # 동점일 경우 후보에 추가

        # 동점인 행동 중 무작위 선택하여 패턴 고착화 방지
        return np.random.choice(best_actions)

    def _get_candidate_moves(self, state, board_size):
        candidates = set()
        occupied = np.argwhere(state != 0)
        for r, c in occupied:
            for dr in range(-4, 5):
                for dc in range(-4, 5):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board_size and 0 <= nc < board_size and state[nr, nc] == 0:
                        candidates.add(nr * board_size + nc)
        return list(candidates)

    def _evaluate_position(self, state, row, col, player):
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        score = 0
        board_size = state.shape[0]

        for dr, dc in directions:
            consecutive = 1
            open_ends = 0

            # 양방향 탐색
            for step in (1, -1):
                r, c = row + dr * step, col + dc * step
                while 0 <= r < board_size and 0 <= c < board_size:
                    if state[r, c] == player:
                        consecutive += 1
                    elif state[r, c] == 0:
                        open_ends += 1
                        break # 빈칸을 만나면 열린 끝으로 간주하고 중단
                    else:
                        break # 상대 돌을 만나면 막힌 것으로 간주하고 중단
                    r += dr * step
                    c += dc * step

            # 패턴별 가중치 (0.0 ~ 1.0 스케일로 정규화 및 우선순위 엄격 유지)
            if consecutive >= 5:
                score += 1.0       # 5목 완성 (승리/패배 직결, 최우선)
            elif consecutive == 4:
                if open_ends == 2: 
                    score += 0.1   # 열린 4목 (다음 턴 무조건 승리)
                elif open_ends == 1: 
                    score += 0.01  # 닫힌 4목 (방어 필수 유도)
            elif consecutive == 3:
                if open_ends == 2: 
                    score += 0.001 # 열린 3목 (공격 전개)
            elif consecutive == 2:
                if open_ends == 2: 
                    score += 0.0001 # 열린 2목 (초반 자리 선점)
                
        return score
    
class KhyAgent:
    def __init__(self, model):
        self.name = "Khy_AI"
        
        # GPU 엔진 설정 (속도 극대화)
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"[시스템] {self.device}")
        
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 1만 판 학습에 맞춘 동적 앱실론 (탐험과 활용의 균형)
        self.gamma = 0.99           # 미래 가치 할인율
        self.epsilon = 1.0          # 초기: 100% 무작위 탐험
        self.epsilon_min = 0.05     # 후반: 최소 5%의 탐험 유지
        self.epsilon_decay = 0.9995 # 감가율 (약 6000판에서 최소치 도달)
        
        # 경험 재생 메모리 (파국적 망각 방지)
        self.memory = deque(maxlen=50000) # 최대 5만 개의 기보 기억
        self.batch_size = 64              # 한 번 복습할 때 꺼내볼 과거 기억의 수


    # 행동 선택 로직
    def select_action(self, state):
        """현재 오목판을 보고 다음 수를 결정합니다."""
        valid_moves = np.where(state.flatten() == 0)[0]
        if len(valid_moves) == 0:
            return 0
        
        # 무작위 탐험 (Epsilon)
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        # 신경망(CNN)의 지식 활용: 가장 점수가 높은 곳 예측
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze()
            
        best_action = valid_moves[0]
        max_q = -float('inf')
        for move in valid_moves:
            if q_values[move].item() > max_q:
                max_q = q_values[move].item()
                best_action = move
                
        return best_action
    
    def get_intrinsic_reward(self, state, action):
        """환경이 주지 않는 칭찬(내적 보상)을 스스로 계산합니다."""
        row, col = divmod(action, 15)
        state_2d = state.reshape(15, 15)
        
        step_reward = 0.0
        
        # 근접 보상: 놓은 자리 주변 8칸에 다른 돌이 하나라도 있는가?
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                r, c = row + dr, col + dc
                
                # 바둑판 범위를 벗어나지 않는지 확인
                if 0 <= r < 15 and 0 <= c < 15:
                    if state_2d[r, c] != 0: # 흑돌(1)이든 백돌(2)이든 존재한다면
                        step_reward += 0.02 # "전투가 벌어지는 곳에 잘 두었어!" 칭찬
                        return step_reward  # 하나만 발견해도 칭찬하고 종료 (중복 칭찬 방지)
                    
        return step_reward # 허허벌판에 두었다면 0점


    # 학습 및 기억 로직 (에피소드 복기 방식)
    def memorize_episode(self, history, final_reward):
        """게임이 끝나면 전체 기보를 평가하여 '오답 노트(메모리)'에 저장만 해둡니다."""
        G = final_reward
        # 게임의 마지막 수부터 첫 수까지 거꾸로 거슬러 올라가며 가치(G)를 매김
        for state, action, step_reward in reversed(history):
            G = step_reward + (self.gamma * G)
            self.memory.append((state, action, G))
            
    def replay_experience(self):
        """저장된 오답 노트에서 무작위로 64개를 뽑아 맹훈련(가중치 업데이트)을 합니다."""
        # 아직 복습할 기억이 64개가 안 모였다면 학습 보류
        if len(self.memory) < self.batch_size:
            return 
            
        # 과거 기억 무작위 추출 (편향 방지)
        mini_batch = random.sample(self.memory, self.batch_size)
        
        states, actions, targets = [], [], []
        for state, action, G in mini_batch:
            states.append(state)
            actions.append(action)
            targets.append(G)
            
        # 64개의 데이터를 한 번에 GPU로 올려서 병렬 연산 (속도 증가)
        states_tensor = torch.FloatTensor(np.array(states)).unsqueeze(1).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        targets_tensor = torch.FloatTensor(targets).to(self.device)
        
        # 모델의 예측값 계산 (64개의 오목판을 동시에 예측)
        q_values = self.model(states_tensor) 
        
        # 64개 결과 중, 내가 실제로 두었던 자리(action)의 예측 점수만 추출
        current_q = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # 오차(예측 점수 vs 실제 정답 G) 계산 및 가중치 업데이트
        loss = nn.MSELoss()(current_q, targets_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()


    # 유틸리티 (모드 전환 및 저장/불러오기)
    def decay_epsilon(self):
        """매 판이 끝날 때마다 탐험 확률을 조금씩 낮춥니다."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def eval_mode(self):
        """실전 대결 모드: 탐험을 완전히 끄고 배운 대로만 둡니다."""
        self.epsilon = 0.0
        self.model.eval()
        
    def train_mode(self):
        """학습 모드: 모델의 신경망을 훈련 상태로 열어둡니다."""
        if self.epsilon == 0.0:
            self.epsilon = 1.0 # 다시 백지상태 탐험부터 시작
        self.model.train()
        
    def save_model(self, file_path="khy_omok_model.pth"):
        """모델의 뇌(가중치)를 파일로 저장합니다."""
        torch.save(self.model.state_dict(), file_path)
        
    def load_model(self, file_path="khy_omok_model.pth"):
        """저장된 뇌(가중치)를 불러와 탑재합니다."""
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path, map_location=self.device, weights_only=True))
            print("[알림] 파일 불러오기 성공")
            self.eval_mode() # 불러온 직후에는 보통 실전을 하므로 모드 전환
        else:
            print("[경고] 파일 불러오기 실패")

# ==========================================
# 3. 대결 실행 루프 (Arena)
# ==========================================
def main():
    env = OmokEnvGUI(render_mode="human")
    agent1 = HumanAgent(env)
    
    
    model = OmokCNN()
    agent2 = KhyAgent(model)
    
    agent2.load_model("khy_omok_model_final.pth")
    agent2.eval_mode()
    
    state, info = env.reset()
    env.render()
    terminated = False
    
    print(f"=== ⚔️ {agent1.name} vs {agent2.name} 대결 시작 ===")
    
    while not terminated:
        # 턴에 따른 상태 반전 논리 (상대는 항상 자신이 흑돌인 것처럼 착각하게 만듦)
        if info["current_player"] == 1:
            start_time = time.time()
            action = agent1.select_action(state)
            end_time = time.time()
            print(f"   ▶ {agent1.name} 착수 완료! (소요 시간: {end_time - start_time:.2f}초)")
        else:
            inverted_state = np.where(state == 1, 2, np.where(state == 2, 1, 0))
            start_time = time.time()
            action = agent2.select_action(inverted_state)
            end_time = time.time()
            print(f"   ▶ {agent2.name} 착수 완료! (소요 시간: {end_time - start_time:.2f}초)")
            
        state, reward, terminated, _, info = env.step(action)
        env.render()
        time.sleep(0.1) # 시각적 확인을 위한 지연

    # 결과 판정
    print("\n=== 🏁 대결 종료 ===")
    winner = info.get("winner")
    if winner == 1: print(f"🎉 {agent1.name} 승리!")
    elif winner == 2: print(f"🎉 {agent2.name} 승리!")
    else: print("🤝 무승부!")
        
    time.sleep(3)
    env.close()

# ===================================================================    
    
# def train_main():
#     env = OmokEnvGUI(render_mode=None) 
    
#     model1 = OmokCNN()
#     agent1 = KhyAgent(model1)
#     agent1.load_model("khy_omok_model_final.pth")
#     agent1.train_mode()
    
#     agent1.epsilon = 0.3
#     agent1.epsilon_decay = 0.9998
    
#     model2 = OmokCNN()
#     agent2 = KhyAgent(model2)
    
#     agent2.model.load_state_dict(agent1.model.state_dict())
#     agent2.eval_mode()
    
#     EPISODES = 10000
#     agent1_wins = 0
#     pbar = tqdm(range(1, EPISODES + 1), desc="학습 진행률")
    
#     for episode in pbar:
#         state, info = env.reset()
#         terminated = False
#         episode_memory = []
        
#         while not terminated:
#             current_player = info["current_player"]
            
#             if current_player == 1:
#                 action = agent1.select_action(state)
#                 # 내적 보상 계산 (이전에 추가하신 함수가 있다면 사용, 없다면 0으로 두셔도 무방합니다)
#                 step_reward = agent1.get_intrinsic_reward(state, action) if hasattr(agent1, 'get_intrinsic_reward') else 0.0
#                 episode_memory.append((state.copy(), action, step_reward))
                
#                 next_state, reward, terminated, _, info = env.step(action)
#                 state = next_state
#             else:
#                 inverted_state = np.where(state == 1, 2, np.where(state == 2, 1, 0))
#                 # 파트너는 agent2의 뇌로 생각해서 둡니다.
#                 action = agent2.select_action(inverted_state) 
#                 next_state, reward, terminated, _, info = env.step(action)
#                 state = next_state

#         # ==========================================
#         # 게임 종료 후: 기억 저장 및 대규모 복습 진행
#         # ==========================================
#         winner = info.get("winner")
#         if winner == 1:
#             final_reward = 1.0     # 승리
#             agent1_wins += 1
#         elif winner == 2:
#             final_reward = -1.0    # 패배
#         else:
#             final_reward = -1.0    # 무승부 (패배 처리)
            
#         # agent1만 학습을 진행합니다. (agent2는 맞으면서 데이터만 제공하는 샌드백 역할)
#         agent1.memorize_episode(episode_memory, final_reward)
#         for _ in range(4):
#             agent1.replay_experience()

#         # 진행 상황 표시 (메모리에 데이터가 얼마나 쌓였는지도 확인 가능)
#         win_rate = (agent1_wins / episode) * 100
#         pbar.set_postfix({
#             "승리": f"{agent1_wins}/{episode} | {win_rate:.2f}%",
#             "앱실론": f"{agent1.epsilon:.3f}",
#             "메모리": f"{len(agent1.memory)}" # 뇌 용량이 차오르는 것을 볼 수 있습니다
#         })
        
#         # 탐험률 감소
#         agent1.decay_epsilon()
        
#         # 1000판마다 중간 저장
#         if episode % 1000 == 0:
#             agent1.save_model(f"khy_omok_model_ep{episode}.pth")
#             agent2.model.load_state_dict(agent1.model.state_dict())
            
#             agent1.epsilon = 0.3
#             agent1.epsilon_decay = 0.9982
            
#     # 전체 학습 종료 후 최종 뇌 구조 저장
#     agent1.save_model("khy_omok_model_final.pth")
#     print("\n=== 1만 판의 수읽기 완료 ===")
#     env.close()

def train_main():
    env = OmokEnvGUI(render_mode=None) 
    
    model1 = OmokCNN()
    agent1 = KhyAgent(model1)
    agent1.load_model("khy_omok_model_final.pth")
    agent1.train_mode()
    
    agent1.epsilon = 0.3
    agent1.epsilon_decay = 0.9982
    
    model2 = OmokCNN()
    agent2 = KhyAgent(model2)
    
    agent2.model.load_state_dict(agent1.model.state_dict())
    agent2.eval_mode()
    
    agent2_heuristic = HeuristicAgent(name="Heuristic_AI")
    current_opponent = agent2
    
    EPISODES = 10000
    agent1_wins = 0
    pbar = tqdm(range(1, EPISODES + 1), desc="학습 진행률")
    
    for episode in pbar:
        if episode == 7001:
            current_opponent = agent2_heuristic
            agent1.epsilon = 0.3
            agent1.epsilon_decay = 0.999
            
        state, info = env.reset()
        terminated = False
        episode_memory = []
        
        while not terminated:
            current_player = info["current_player"]
            
            if current_player == 1:
                action = agent1.select_action(state)
                # 내적 보상 계산 (이전에 추가하신 함수가 있다면 사용, 없다면 0으로 두셔도 무방합니다)
                step_reward = agent1.get_intrinsic_reward(state, action) if hasattr(agent1, 'get_intrinsic_reward') else 0.0
                episode_memory.append((state.copy(), action, step_reward))
                
                next_state, reward, terminated, _, info = env.step(action)
                state = next_state
            else:
                inverted_state = np.where(state == 1, 2, np.where(state == 2, 1, 0))
                # 파트너는 agent2의 뇌로 생각해서 둡니다.
                action = current_opponent.select_action(inverted_state)
                next_state, reward, terminated, _, info = env.step(action)
                state = next_state

        # ==========================================
        # 게임 종료 후: 기억 저장 및 대규모 복습 진행
        # ==========================================
        winner = info.get("winner")
        if winner == 1:
            final_reward = 1.0     # 승리
            agent1_wins += 1
        elif winner == 2:
            final_reward = -1.0    # 패배
        else:
            final_reward = -1.0    # 무승부 (패배 처리)
            
        # agent1만 학습을 진행합니다. (agent2는 맞으면서 데이터만 제공하는 샌드백 역할)
        agent1.memorize_episode(episode_memory, final_reward)
        for _ in range(4):
            agent1.replay_experience()

        # 진행 상황 표시 (메모리에 데이터가 얼마나 쌓였는지도 확인 가능)
        win_rate = (agent1_wins / episode) * 100
        pbar.set_postfix({
            "승리": f"{agent1_wins}/{episode} | {win_rate:.2f}%",
            "앱실론": f"{agent1.epsilon:.3f}",
            "메모리": f"{len(agent1.memory)}" # 뇌 용량이 차오르는 것을 볼 수 있습니다
        })
        
        # 탐험률 감소
        agent1.decay_epsilon()
        
        # 1000판마다 중간 저장
        if episode % 1000 == 0:
            agent1.save_model(f"khy_omok_model_ep{episode}.pth")
            if episode <= 7000:
                agent2.model.load_state_dict(agent1.model.state_dict())
                agent1.epsilon = 0.3
                agent1.epsilon_decay = 0.9982
            
    # 전체 학습 종료 후 최종 뇌 구조 저장
    agent1.save_model("khy_omok_model_final.pth")
    print("\n=== 1만 판의 수읽기 완료 ===")
    env.close()
    
# ==========================================
# 4. 메인
# ==========================================
if __name__ == "__main__":
    main()
    # train_main()