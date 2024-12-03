import os
import argparse

# task_type = 'dummy_drawer'
arm_type = 'arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='all')
    args = parser.parse_args()
    
    if args.model_type == 'all':
        model_types = sorted(os.listdir('./results'))
    else:
        model_types = [args.model_type]
    
    model_accuracy_dict = {}
    
    for model in model_types:
        
        model_accuracy_dict[model] = {}
        for task_type in sorted(os.listdir(os.path.join('./results', model))):
            
            # if task_type in ['frl_apartment_stage_simple', 'modern_bedroom_no_roof', 'modern_office_no_roof']:
            #     continue
        
            for env in sorted(os.listdir(os.path.join('./results', model, task_type, arm_type))):
                
                if 'Drawer' in env:
                    task = env[:env.index('Drawer') + len('Drawer')]
                elif 'Grasp' in env:
                    task = env[:env.index('Can') + len('Can')]
                elif 'Near' in env:
                    task = env[:env.index('Near') + len('Near')]
                
                model_accuracy_dict[model][task] = model_accuracy_dict[model].get(task, [])
                
                for trial in sorted(os.listdir(os.path.join('./results', model, task_type, arm_type, env))):
                    
                    vid_names = [f for f in os.listdir(os.path.join('./results', model, task_type, arm_type, env, trial)) if '.mp4' in f]
                    
                    for vid_name in vid_names:
                        if 'success' in vid_name:
                            model_accuracy_dict[model][task].append(1)
                        else:
                            model_accuracy_dict[model][task].append(0)

    num_completed = 0
    results_dict = {}
    for model in model_accuracy_dict:
        
        for task in model_accuracy_dict[model]:
            
            results_dict[task] = sum([1 for i in model_accuracy_dict[model][task] if i == 1]) / len(model_accuracy_dict[model][task])
    
    print()
    for task in results_dict:
        print(f'{task}: {results_dict[task]}, {len(model_accuracy_dict[model][task])} trials')
        num_completed += len(model_accuracy_dict[model][task])
        
    print('Total number of completed trials:', num_completed)