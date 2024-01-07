# Sim baslat
roslaunch envsim visionenv_sim.launch render:=True
# Using the GUI, press Arm & Start to take off.

# Evaluation 
python evaluation_node.py

# Kontrol komutlari uretecek olan kodu calistir
cd envtest/ros
python run_competition.py [--vision_based]
python run_competition.py --ppo_path=/home/gazi13/catkin_ws_agile/src/agile_flight/envtest/python/saved/PPO_1


# Sistem bu mesajı almadan kontrol komutlarini dinlemez - START komutu gibi 
rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" -1

# Bütün bu adımları otomatik yapması icinde bir bash yazmislar
launch_evaluation.bash N

# Sim ortamını resetlemek için
rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}"
--------------------------------------------------------
# TRAIN
cd /home/gazi13/catkin_ws_agile/src/agile_flight/envtest/
python3 -m python.run_vision_ppo --render 0 --train 1


# INFERENCE 
# Benim denediğim komut
roslaunch envsim visionenv_sim.launch render:=True
cd /home/gazi13/catkin_ws_agile/src/agile_flight/envtest/ros
python run_competition.py --ppo_path=/home/gazi13/catkin_ws_agile/src/agile_flight/envtest/python/saved/PPO_18
rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" -1

# Dökümantasyon içerisindeki komut - trial_num kacinci egitim oldugu - iter de iterasyon numarasi
python3 -m python.run_vision_ppo --render 0 --train 0 --trial trial_num --iter iter_num 
python3 -m python.run_vision_ppo --render 1 --train 0 --trial 00 --iter 500 



----------------------------------------
# Inference FAST
python3 -m python.run_vision_ppo --render 1 --train 0 --trial 38 --iter 550 // bu komut ile run_competition farkli sonuç veriyor !!!!
rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" -1
rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}"


---------------

tensorboard --logdir=./

------------------------------------ AGILE ENVIRONMEN ------------------------------------

cd /home/gazi13/catkin_ws_agile/src/agile_flight/envtest/
python3 -m python.run_agile_ppo --render 0 --train 1
