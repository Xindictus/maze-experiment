Installation for windows

```
python -m venv env

env\Scripts\activate

pip install -r requirements.txt
```


Installation for ubuntu

```
python3 -m venv env

source env/bin/activate

pip install -r requirements.txt
```

Then use one of the commands bellow. Old version of Maze unity may have some bugs. 


---
No tl collaboration (Collaboration between human and Ai agent)

python sac_maze3d_train.py --config game\config\config_sac_No_TL.yaml --participant Name --num-actions 3 --auto-alpha

---

One agent controlling both axes (One agents plays alone)

python sac_maze3d_train.py --config game\config\config_sac_No_TL_agent_only.yaml --participant Name --num-actions 9 --auto-alpha

Two agent (Collaboration between two agents)

python sac_maze3d_train.py --config game\config\config_sac_No_TL_Two_agents.yaml --participant Name --num-actions 3 --auto-alpha

