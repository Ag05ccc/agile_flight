import os

policy_path = "saved/PPO_1/Policy/"

# Get last generated policy
current_dir = os.path.dirname(os.path.abspath(__file__))
trial_num = os.path.join(current_dir, policy_path)

print("current_dir : ",current_dir)
print("trial_num   : ",trial_num)
# Ensure the directory exists
if not os.path.exists(trial_num):
    print(f"Directory {trial_num} does not exist.")
else:
    # List all files in the directory with their full paths
    iters_list = [os.path.join(trial_num, f) for f in os.listdir(trial_num)]
    policy_dir = max(iters_list, key=os.path.getctime)
    rms_dir = os.sep.join(policy_dir.split(os.sep)[:-2]) +"/RMS/"+policy_dir[:]


print("policy_dir : ",policy_dir)
print("rms_dir    : ",rms_dir)