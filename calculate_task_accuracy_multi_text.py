import os
import argparse

# task_type = 'dummy_drawer'
arm_type = 'arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner'
non_augmented_tasks = ['close bottom drawer', 'open bottom drawer',
                       'open top drawer', 'close top drawer', 'close middle drawer', 'open middle drawer',
                       'grab the coke', 'grasp the coke', 'lift the coke', 'obtain the coke', 'pick the coke',
                       'pick up the coke', 'retrieve the coke', 'take the coke', 'bring apple close to coke',
                       'bring apple near coke', 'move apple close to coke', 'place apple adjacent to coke',
                       'place apple close to coke', 'place apple near coke', 'put apple next to coke',
                       'put apple near coke']

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
            
            if task_type in ['frl_apartment_stage_simple', 'modern_bedroom_no_roof', 'modern_office_no_roof']:
                continue
        
            for env in sorted(os.listdir(os.path.join('./results', model, task_type, arm_type))):
                
                if 'Drawer' in env:
                    task = env[:env.index('Drawer') + len('Drawer')]
                elif 'Grasp' in env:
                    task = env[:env.index('Can') + len('Can')]
                elif 'Near' in env:
                    task = env[:env.index('Near') + len('Near')]
                
                model_accuracy_dict[model][task] = model_accuracy_dict[model].get(task, {})
                
                for language_task in os.listdir(os.path.join('./results', model, task_type, arm_type, env)):
                    
                    model_accuracy_dict[model][task][language_task] = model_accuracy_dict[model][task].get(language_task, [])
                
                    for trial in sorted(os.listdir(os.path.join('./results', model, task_type, arm_type, env, language_task))):
                        
                        vid_name = [f for f in os.listdir(os.path.join('./results', model, task_type, arm_type, env, language_task, trial)) if '.mp4' in f]
                         
                        if len(vid_name) > 0:
                            vid_name = vid_name[0]
                        else:
                            continue
                        if 'success' in vid_name:
                            model_accuracy_dict[model][task][language_task].append(1)
                        else:
                            model_accuracy_dict[model][task][language_task].append(0)

    num_completed = 0
    results_dict = {}
    for model in model_accuracy_dict:
        
        for task in model_accuracy_dict[model]:
            
            for lang_inst in model_accuracy_dict[model][task]:
                
                results_dict[task] = results_dict.get(task, {})
                
                if lang_inst in non_augmented_tasks:
                    results_dict[task]['non_augmented'] = results_dict[task].get('non_augmented', [])
                    results_dict[task]['non_augmented'].extend(model_accuracy_dict[model][task][lang_inst])
                else:
                    results_dict[task]['augmented'] = results_dict[task].get('augmented', [])
                    results_dict[task]['augmented'].extend(model_accuracy_dict[model][task][lang_inst])
                    
                # print(model, task, lang_inst, sum(model_accuracy_dict[model][task][lang_inst]) / len(model_accuracy_dict[model][task][lang_inst]), len(model_accuracy_dict[model][task][lang_inst]))
                num_completed += len(model_accuracy_dict[model][task])
    
    print()
    for task in results_dict:
        print('Task:', task)
        
        if 'augmented' in results_dict[task]:
            print('Augmented:', sum(results_dict[task]['augmented']) / len(results_dict[task]['augmented']), len(results_dict[task]['augmented']))
        
        if 'non_augmented' in results_dict[task]:
            print('Non-augmented:', sum(results_dict[task]['non_augmented']) / len(results_dict[task]['non_augmented']), len(results_dict[task]['non_augmented']))
        print()
        
    print('Total number of completed trials:', num_completed)