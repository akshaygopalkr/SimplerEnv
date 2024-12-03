# !/bin/bash

cp1="/data/shresth/openvla-checkpoints/prism-dinosiglip-224px+mx-rt_1_4_3d_string_grounded/checkpoints/step-295000-epoch-24-loss=0.0845.pt"

# Run opendvla evaluation
# bash scripts/opendvla_drawer_visual_matching.sh openvla $cp1 
bash scripts/openvla_move_near_variant_agg.sh openvla $cp1
bash scripts/openvla_move_near_visual_matching.sh openvla $cp1
bash scripts/openvla_pick_coke_can_variant_agg.sh openvla $cp1
bash scripts/openvla_pick_coke_can_visual_matching.sh openvla $cp1
# bash scripts/opendvla_put_in_drawer_variant_agg.sh openvla $cp1
# bash scripts/opendvla_put_in_drawer_visual_matching.sh openvla $cp1
# bash scripts/opendvla_drawer_variant_agg.sh openvla $cp1

# bash scripts/opendvla_move_near_variant_agg.sh openvla $cp2
# bash scripts/opendvla_move_near_visual_matching.sh openvla $cp2
# bash scripts/opendvla_pick_coke_can_variant_agg.sh openvla $cp2
# bash scripts/opendvla_pick_coke_can_visual_matching.sh openvla $cp2
# bash scripts/opendvla_put_in_drawer_variant_agg.sh openvla $cp2
# bash scripts/opendvla_put_in_drawer_visual_matching.sh openvla $cp2
# bash scripts/opendvla_drawer_variant_agg.sh openvla $cp2
# bash scripts/opendvla_drawer_visual_matching.sh openvla $cp2

# Run openvla evaluation
# bash scripts/opendvla_drawer_visual_matching.sh openvla None
# bash scripts/opendvla_move_near_variant_agg.sh openvla $gr
# bash scripts/opendvla_move_near_visual_matching.sh openvla $gr
# bash scripts/opendvla_pick_coke_can_variant_agg.sh openvla None
# bash scripts/opendvla_pick_coke_can_visual_matching.sh openvla None
# bash scripts/opendvla_put_in_drawer_variant_agg.sh openvla None
# bash scripts/opendvla_put_in_drawer_visual_matching.sh openvla None
# bash scripts/opendvla_drawer_variant_agg.sh openvla None
