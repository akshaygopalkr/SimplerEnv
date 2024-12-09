# !/bin/bash

cp1="/data/shresth/openvla-checkpoints/prism-dinosiglip-224px+mx-rt_1_text_aug_4/checkpoints/step-240000-epoch-08-loss=0.1789.pt"
cp2="/data/shresth/openvla-checkpoints/prism-dinosiglip-224px+mx-rt_1_all_aug_4/checkpoints/step-470000-epoch-15-loss=0.1603.pt"

# Run opendvla evaluation
bash scripts/openvla_drawer_visual_matching.sh $cp1 
# bash scripts/openvla_move_near_variant_agg.sh $cp1
bash scripts/openvla_move_near_visual_matching.sh $cp1
# bash scripts/openvla_pick_coke_can_variant_agg.sh $cp1
bash scripts/openvla_pick_coke_can_visual_matching.sh $cp1
# bash scripts/openvla_put_in_drawer_variant_agg.sh $cp1
bash scripts/openvla_put_in_drawer_visual_matching.sh $cp1
# bash scripts/openvla_drawer_variant_agg.sh $cp1

# bash scripts/openvla_move_near_variant_agg.sh $cp2
bash scripts/openvla_move_near_visual_matching.sh $cp2
# bash scripts/openvla_pick_coke_can_variant_agg.sh $cp2
bash scripts/openvla_pick_coke_can_visual_matching.sh $cp2
# bash scripts/openvla_put_in_drawer_variant_agg.sh $cp2
bash scripts/openvla_put_in_drawer_visual_matching.sh $cp2
# bash scripts/openvla_drawer_variant_agg.sh $cp2
bash scripts/openvla_drawer_visual_matching.sh $cp2

# Run openvla evaluation
# bash scripts/openvla_drawer_visual_matching.sh None
# bash scripts/openvla_move_near_variant_agg.sh None
# bash scripts/openvla_move_near_visual_matching.sh None
# bash scripts/openvla_pick_coke_can_variant_agg.sh None
# bash scripts/openvla_pick_coke_can_visual_matching.sh None
# bash scripts/openvla_put_in_drawer_variant_agg.sh None
# bash scripts/openvla_put_in_drawer_visual_matching.sh None
# bash scripts/openvla_drawer_variant_agg.sh None
