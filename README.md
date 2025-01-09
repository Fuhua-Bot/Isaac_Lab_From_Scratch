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

## 🎲 Random Agent Test Environment
Test the action space of the environment with random actions using the following command:

```bash
isaacpython scripts/random_agents.py

## 🏋️ Start Reinforcement Learning Training
Start training using the following command:

```bash
isaaclabpython rsl_rl/train.py --task=project_name --num_envs=4096 --headless
Parameter Explanation:
--task: Specifies the task name, which should match the project name.
--num_envs: Sets the number of parallel environments (e.g., 4096).
--headless: Runs in headless mode (without a graphical interface).
--resume: Resumes training from a specified log path, for example:


```bash
logs/rsl_rl/project_name/2025-01xxxxx/

## ⚙️ Overview of Scripts
rename_template.py
Used for renaming projects.

Usage:

```bash
isaacpython scripts/rename_template.py project_name
