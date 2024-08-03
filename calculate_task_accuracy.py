import os

task_type = 'dummy_drawer'
arm_type = 'arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner'

if __name__ == '__main__':
    
    model_accuracy_dict = {}
    
    for model in sorted(os.listdir('./results')):
        
        model_accuracy_dict[model] = {}
        
        for env in sorted(os.listdir(os.path.join('./results', model, task_type, arm_type))):
            
            task = env[:env.index('Drawer') + len('Drawer')]
            model_accuracy_dict[model][task] = model_accuracy_dict[model].get(task, [])
            
            for trial in sorted(os.listdir(os.path.join('./results', model, task_type, arm_type, env))):
                
                vid_name = [f for f in os.listdir(os.path.join('./results', model, task_type, arm_type, env, trial)) if '.mp4' in f][0]
                if 'success' in vid_name:
                    model_accuracy_dict[model][task].append(1)
                else:
                    model_accuracy_dict[model][task].append(0)

    for model in model_accuracy_dict:
        
        for task in model_accuracy_dict[model]:
            
            print(model, task, sum(model_accuracy_dict[model][task]) / len(model_accuracy_dict[model][task]))
                
            