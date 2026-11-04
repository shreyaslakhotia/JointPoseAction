# Synthetic implementation validation

Known virtual primitives only; no Assembly101 or learned reconstruction result.

| Method | Partition | Reference | Available views | IoU | Dice | Boundary F1 |
|---|---|---|---:|---:|---:|---:|
| box_16 | input | prediction agreement | 0 | unavailable | unavailable | unavailable |
| box_16 | input | independent reference | 6 | 0.8299 | 0.9069 | 0.6313 |
| box_16 | heldout | prediction agreement | 0 | unavailable | unavailable | unavailable |
| box_16 | heldout | independent reference | 2 | 0.8468 | 0.9170 | 0.6963 |
| box_32 | input | prediction agreement | 0 | unavailable | unavailable | unavailable |
| box_32 | input | independent reference | 6 | 0.9125 | 0.9540 | 0.9768 |
| box_32 | heldout | prediction agreement | 0 | unavailable | unavailable | unavailable |
| box_32 | heldout | independent reference | 2 | 0.9607 | 0.9799 | 1.0000 |
| sphere_16 | input | prediction agreement | 0 | unavailable | unavailable | unavailable |
| sphere_16 | input | independent reference | 6 | 0.8515 | 0.9198 | 0.8747 |
| sphere_16 | heldout | prediction agreement | 0 | unavailable | unavailable | unavailable |
| sphere_16 | heldout | independent reference | 2 | 0.7569 | 0.8616 | 0.5178 |
| sphere_32 | input | prediction agreement | 0 | unavailable | unavailable | unavailable |
| sphere_32 | input | independent reference | 6 | 0.8807 | 0.9365 | 0.9820 |
| sphere_32 | heldout | prediction agreement | 0 | unavailable | unavailable | unavailable |
| sphere_32 | heldout | independent reference | 2 | 0.8174 | 0.8995 | 0.7517 |
