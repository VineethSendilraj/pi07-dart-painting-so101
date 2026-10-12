116 teleop demos of an **SO-101 arm aiming a motorized dart blaster at a 5×5 board**. Each episode is one shot. The task string is `"shoot <cell>"` (A1 top left … E5 bottom right). We recorded them at the Actor Labs × Physical Intelligence Act-athon (Sep 2026, 3rd place) and fine-tuned **π0.7** on them. Claude picks the cells, the policy aims and fires, and together they draw pictures with darts.

- **Code, hardware and wiring:** https://github.com/VineethSendilraj/pi07-dart-painting-so101
- **Demo video and write-up:** https://x.com/agno_three/status/2105353980377546994
- **One-page write-up (PDF):** https://github.com/VineethSendilraj/pi07-dart-painting-so101/blob/main/docs/report/report.pdf
- **Authors:** Vineeth Sendilraj ([@VineethSendil](https://x.com/VineethSendil)), Vardhan Agnihotri ([@agno_three](https://x.com/agno_three)), Yuvraj Lakhotia ([@yuvrajlakh](https://x.com/yuvrajlakh))

### Contents

116 episodes, 29,131 frames at 30 fps (~16 min), all marked good when reviewed. 23 cells are covered: every cell except A5 and E5.

| Key | Shape | Contents |
| --- | --- | --- |
| `observation.images.front` | 480×640×3 | rear camera: arm, blaster and the whole board |
| `observation.images.wrist` | 480×640×3 | gun camera, mounted above the barrels |
| `observation.state` | 5 | follower joints in degrees: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll |
| `action` | 6 | leader joints in degrees (same order) + `trigger`: 1 for the 10 steps after the fire button, else 0 |

The arm has no gripper. The blaster sits on a tripod and the arm moves its handle.

`meta/blaster_episodes.jsonl` gives these fields for each episode:
- `source_episode_id`
- `cell`
- `live`: 23 takes fired a real dart. The other 93 were dry, with the same motion and the master switch off.
- `board`: clutter level, `light` / `heavy` / `late`.
- `start`: `rest`, or the cell the arm started aimed at.

### Things to know

- **Post-shot tail.** Every take keeps recording 2 s after the shot, with the arm still on target and `trigger=0`. Trim after the trigger pulse if you want "fire" to dominate the on-target frames.
- **Crosshair overlay.** 102 of 116 takes have the recording app's red crosshair burned into the wrist image. The earliest takes (0000–0014) have none or a different mark.
- **Uneven coverage.** Row 1, A3 and E3 have 2 takes each. A5 and E5 have none.
- **Fixed rig.** The camera and board positions are fixed, so aim is close to a fixed pan/tilt per cell.

### License and attribution

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): use, share and adapt freely, including commercially, as long as you credit the authors and link to this dataset. If you use it in a paper, please cite it with the BibTeX in the Citation section.
