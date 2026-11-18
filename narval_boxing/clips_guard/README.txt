Guard-bracketed boxing clips

Every clip now starts and ends in the guard pose so they can be chained later.
Before and after each move there is a short guard hold (0.2 s) and a smooth
blend (0.3-0.6 s). The body stays where the move ended; only the posture goes
back to guard.

Clips (MyoFullBody/gmr_v6/skills/):
  guard added:     E19_dodge_left, E19_dodge_left_t2, flurry_kick, kick_R,
                   lunge_push, recover_jump_landing, recover_kick,
                   recover_lunge, run_fwd
  already guard:   guard, punch_L_jumpinplace1, punch_R_jumpinplace1

Left out (need fixing first):
  E1_turn_around_right, E2_turn_around_left, E5_hook_left, E6_hook_right
    arm joints are about 100 deg past the model's joint limits; re-retarget
    with the same GMR pipeline as the other clips.

Known issues:
  recover_jump_landing starts mid-air, so the lead-in floats into the jump;
    re-cut it to include the takeoff.
  E19_dodge_left has one glitchy frame (E19_dodge_left_t2 is clean).

Videos: in guard_clips_for_team.zip (not committed). all_clips.mp4 plays all 12;
yellow label = added guard/blend, white label = original clip.

Use for training: point CONVERTED_AMASS_PATH at narval_boxing/clips_guard (the one that
contains MyoFullBody/) and keep rel_dataset_path entries as skills/<name>.

Regenerate after fixing clips (from the musclemimic_mcgill_dev repo):
  .venv/bin/python <path to myosuite-capstone>/scripts/bracket_clips_with_guard.py \
      --src_dir <clips>/MyoFullBody/gmr_v6/skills \
      --dst_dir <out>/MyoFullBody/gmr_v6/skills
