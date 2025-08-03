## USE THIS AURORA ONE LINER TO SET UP THE SHADOW HAND ENVIROMENT

```bash
bash <(curl -Ls https://raw.githubusercontent.com/shadow-robot/aurora/v2.2.5/bin/run-ansible.sh) docker_deploy \
  --branch=v2.2.5 \
  --inventory=local \
  product=hand_e \
  tag=noetic-v1.0.31 \
  reinstall=true \
  sim_icon=true \
  sim_hand=true \
  container_name=dexterous_hand_simulated
```
