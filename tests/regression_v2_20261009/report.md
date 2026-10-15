# fire_smoke_v2.pt regression — 2026-10-09

Operating point: conf 0.35, imgsz 640, fire gate 1.5 s, smoke gate 2.5 s.

## Videos
| folder | video | expected | fire @2fps | smoke @2fps | fire @5fps | smoke @5fps | result |
| orange | Cars_passing_orange_lit_building_20261009120454.mp4 | NONE | none | none | none | none | PASS |
| orange | Empty_indoor_corridor_CCTV_footage_20261009115953.mp4 | NONE | none | none | none | none | PASS |
| orange | Person_walking_with_orange_jacket_20261009121628.mp4 | NONE | none | none | none | none | PASS |
| orange | Workers_walking_on_factory_floor_20261009120333.mp4 | NONE | none | none | none | none | PASS |
| orange | Workers_walking_past_warehouse_wall_20261009120155.mp4 | NONE | none | none | none | none | PASS |
| smoke | Fire_burning_in_industrial_yard_20261009130200.mp4 | FIRE | 1.0s | none | 1.46s | none | PASS |
| smoke | Fire_burning_in_waste_bin_20261009130018.mp4 | FIRE | 1.0s | none | 1.46s | none | PASS |
| smoke | Smoke_rising_behind_cardboard_boxes_20261009130315.mp4 | SMOKE | none | none | none | none | MISS |
| smoke | Smoke_rising_in_parking_lot_20261009125826.mp4 | SMOKE | none | 2.5s | none | 3.75s | PASS |
| smoke | Smoke_rising_in_parking_lot_20261009130400.mp4 | SMOKE | none | 2.5s | none | 2.71s | PASS |
| smoke | Smoke_rising_in_warehouse_corner_20261009125643.mp4 | SMOKE | none | 6.5s | none | 6.88s | PASS |
| smoke | Steam_venting_from_kitchen_20261009130204.mp4 | NONE | none | 4.0s | none | 4.17s | FALSE ALARM |

## Labeled benchmark @0.35 (240 images)
| class | recall | precision | tp / fp / fn |
| fire | 0.773 | 1.0 | 92 / 0 / 27 |
| smoke | 0.887 | 0.954 | 125 / 6 / 16 |

## Totals
PASS 10, MISS 1, FALSE ALARM 1, UNKNOWN 0
