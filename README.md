# 🌟 Isaac_Lab_From_Scratch

Welcome to **Isaac_Lab_From_Scratch**, a repository designed to help you develop robotic reinforcement learning (RL) environments in Isaac Lab from scratch. 🤖

---

## ✨ Key Features

- 🛠️ **Develop robotic RL environments** from scratch.
- 🎮 Integration with **NVIDIA Isaac Sim** for realistic robot simulations.
- 🧩 Tools for validating robot assembly, degrees of freedom (DOF), and action spaces.
- 🏗️ Modular design for seamless customization of robots and environments.

---

## 📋 Prerequisites

To use this repository, ensure the following are installed and configured:

1. 💻 **Isaac Sim**: [Download Isaac Sim](https://developer.nvidia.com/isaac-sim).
2. ⚙️ **Isaac Lab**: Properly configure Isaac Lab for environment development.
3. 🖥️ **IDE**: It is recommended to use [Visual Studio Code (VSCode)](https://code.visualstudio.com/).

---

## 🚀 Development Workflow

### 🛠️ Step 1: CAD Design to USD File
- ✏️ Design your robot in **Onshape** (or other CAD tools).
- 📂 Export the robot model as a **USD file**.

### 🔎 Step 2: Verify in Isaac Sim
- 🚢 Import the USD file into **Isaac Sim**.
- ✅ Verify the assembly and degrees of freedom (DOF) in Isaac Sim.

### 📜 Step 3: Create Robot Configuration (`robot cfg`)
- 📝 Write a robot configuration file (e.g., `robot_cfg.py`).
- 🧪 Use the `display.py` script to:
  - ✅ Validate that the robot is correctly loaded.
  - 🔗 Verify joint indices.

### ⚙️ Step 4: Define Environment Configuration (`env cfg`)
- ✍️ Write your RL environment configuration.
- 🎲 Use `random_agents.py` to:
  - 🧪 Test the action space with random actions.
  - 🛡️ Ensure your configuration is correct.

### 🏋️ Step 5: Train the Model
- 🎓 Begin RL training using the `rsl_rl/train.py` script.

---

## 🗂️ Repository Structure

```plaintext
Isaac_Lab_From_Scratch/
├── bip_wl_agents/         # 🤖 RL agent configurations and scripts
├── bip_wl_mdp/            # 🔄 MDP components: observations, rewards, terminations
├── bip_wl_robot/          # 🛠️ Robot configurations and USD assets
├── bip_wl_tasks/          # 📂 Task-specific environments and configurations
├── scripts/               # ⚙️ Utility scripts for development and validation
├── README.md              # 📜 Project documentation
├── LICENSE                # ⚖️ License information
├── pyproject.toml         # 🐍 Python project configuration
└── requirements.txt       # 📦 Python dependencies
