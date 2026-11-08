# APP_sim With leakage - CSV glossary

One row every 0.5 deg of shaft rotation (720 points per revolution), revolutions 1-3 from a cold start (p_initial everywhere). Frame and signs: right-handed (+z towards the valve plate, shaft angle clockwise seen from +z). Pressures in bar absolute unless stated; leakage + = out of the chamber.

## Prefixes

| Prefix | Meaning |
|---|---|
| `P1_` | Piston / chamber 1 (piston 1 is at the shaft angle phi; piston j trails it by (j-1)*40 deg). |
| `ALL_` | Sum over the 9 pistons. |
| `Pd, Ps` | Discharge (delivery line) and suction line pressure. |
| `Q_` | Pump-level flows. |
| `eta_vol` | Volumetric efficiency. |
| `T_` | Shaft torques. |
| `BLK_` | Cylinder block, external loads (from the pistons, spring, bearing). |
| `SW_` | Swash plate. |
| `SL1_ / SL_` | Slipper 1 / all 9 slippers (slipper-swash plate gap). |
| `VP_` | Cylinder block - valve plate gap. VP_K1 = kidney of chamber 1. |
| `PB1_ / PB_` | Piston 1 / all 9 pistons, piston-bore gap. |
| `LEAK_` | Leakage summed over all gaps. |

## Unit suffixes

| Suffix | Unit |
|---|---|
| `_s` | seconds |
| `_deg` | degrees |
| `_bar` | bar (absolute pressure) |
| `_LPM` | litres per minute |
| `_pct` | percent |
| `_mm` | millimetres |
| `_mm3` | cubic millimetres |
| `_um` | micrometres |
| `_mrad` | milliradians |
| `_ms` | metres per second |
| `_ms2` | metres per second squared |
| `_N` | newtons |
| `_Nm` | newton-metres |

## Short terms

| Term | Meaning |
|---|---|
| `F_DK` | Pressure force on the piston (Ivantysyn D = Druck, K = Kolben/piston). |
| `F_aK` | Inertia force of the reciprocating piston + slipper. |
| `F_AK` | Total axial force on the piston, F_DK + F_aK. |
| `F_SK` | Swash-plate (S) force on the piston (K) along the plate normal n. |
| `F_TG` | Friction (T) at the slipper (G = Gleitschuh). |
| `F_wK` | Centrifugal force (omega) of the piston + slipper. |
| `F_RK` | Radial force on the piston at the ball joint. |
| `arm` | Cradle moment arm: y-component of (contact point - cradle axis) x n. |
| `h0` | Film thickness at the centre (slipper, valve plate). |
| `hmin` | Minimum film thickness over the gap. |
| `xi, zeta` | Slipper frame: xi = sliding direction, zeta = n x xi. |
| `r, t` | Piston frame: r = radially outward from the shaft axis, t = z x r. |
| `case / ch` | Piston-bore film ends: bore exit at the block front (case pressure) / piston end face (chamber). |
| `zeta (piston)` | Axial coordinate along the piston, 0 at the end face, negative towards the bore exit. |
| `leak` | Leakage, + = out of the chamber, in L/min (LPM). |
| `ext / fluid / contact` | External load / fluid film force (incl. squeeze) / asperity contact penalty force. |

## Columns


### run

| # | Column | Unit | Meaning |
|---|---|---|---|
| 1 | `time_s` | s | Time since the cold start. |
| 2 | `revolution` | - | Shaft revolution number (1 = first revolution after the cold start). |
| 3 | `phi_abs_deg` | deg | Shaft angle since the start, 0-1080 deg over three revolutions (clockwise seen from +z, simulation frame). |
| 4 | `phi_deg` | deg | Shaft angle within the revolution, 0-360 deg. Piston 1 sits at this angle (phi = 0 near BDC; BDC ~351 deg, TDC ~171 deg). |

### hydraulics

| # | Column | Unit | Meaning |
|---|---|---|---|
| 5 | `P1_p_bar` | bar | Chamber pressure of piston 1 (absolute). |
| 6 | `P1_Q_port_LPM` | L/min | Flow of chamber 1 through its valve-plate kidney, + out of the chamber into the ports (discharge), - into the chamber (suction). |
| 7 | `Pd_bar` | bar | Discharge (delivery line) pressure (absolute). |
| 8 | `Ps_bar` | bar | Suction line pressure (absolute). |
| 9 | `Q_discharge_LPM` | L/min | Discharge flow: sum of all kidney flows into the delivery line. |
| 10 | `Q_pump_LPM` | L/min | Pump flow: flow through the load orifice (what the pump delivers to the system). |
| 11 | `Q_geometric_LPM` | L/min | Geometric flow 9 * Ak * stroke * n (no leakage, incompressible). |
| 12 | `eta_vol_inst_pct` | % | Instantaneous volumetric efficiency Q_pump / Q_geometric. |
| 13 | `eta_vol_rev_pct` | % | Volumetric efficiency of the revolution: revolution-mean Q_pump / Q_geometric. |

### kinematics

| # | Column | Unit | Meaning |
|---|---|---|---|
| 14 | `P1_sk_mm` | mm | Piston 1 stroke position from BDC towards the valve plate (+z). |
| 15 | `P1_vz_ms` | m/s | Piston 1 axial velocity v_z (+ towards the valve plate = discharge stroke). |
| 16 | `P1_az_ms2` | m/s^2 | Piston 1 axial acceleration a_z. |
| 17 | `P1_V_mm3` | mm^3 | Chamber 1 volume. |
| 18 | `P1_lF_mm` | mm | Piston 1 engagement (guide) length in the bore, block front face to piston end face. |
| 19 | `P1_arm_mm` | mm | Piston 1 cradle moment arm: cradle torque = normal force * arm. |

### torque

| # | Column | Unit | Meaning |
|---|---|---|---|
| 20 | `P1_T_pressure_Nm` | N.m | Piston 1 cradle torque from chamber pressure, F_DK / c * arm (Tpp). + = CCW seen from +y. |
| 21 | `P1_T_inertia_Nm` | N.m | Piston 1 cradle torque from piston inertia, F_aK / c * arm (Tpi). |
| 22 | `P1_T_cradle_Nm` | N.m | Piston 1 cradle torque, pressure + inertia. |
| 23 | `ALL_T_pressure_Nm` | N.m | Cradle torque from pressure, sum of the 9 pistons. |
| 24 | `ALL_T_inertia_Nm` | N.m | Cradle torque from inertia, sum of the 9 pistons. |
| 25 | `ALL_T_cradle_Nm` | N.m | Total cradle torque of the 9 pistons (pressure + inertia). |
| 26 | `T_drive_Nm` | N.m | Input shaft (drive) torque M_z from the piston radial forces. |
| 27 | `T_drive_ideal_Nm` | N.m | Ideal drive torque V_g (Pd - Ps) / (2 pi). |

### piston forces

| # | Column | Unit | Meaning |
|---|---|---|---|
| 28 | `P1_F_DK_N` | N | Pressure force on piston 1, -(p - p_case) Ak along z. |
| 29 | `P1_F_aK_N` | N | Inertia force of piston 1 + slipper, -m_k a_z along z. |
| 30 | `P1_F_AK_N` | N | Total axial force F_DK + F_aK. |
| 31 | `P1_F_SK_N` | N | Swash-plate normal force of piston 1, + = pressing on the plate (F_SK = F_AK / c, sign flipped). |
| 32 | `P1_F_SKx_N` | N | Swash-plate reaction on piston 1 at the ball joint, x component. |
| 33 | `P1_F_SKy_N` | N | Swash-plate reaction on piston 1 at the ball joint, y component. |
| 34 | `P1_F_TG_N` | N | Slipper viscous friction magnitude (opposes the sliding velocity). |
| 35 | `P1_F_wK_N` | N | Centrifugal force of piston 1 + slipper, m_k w^2 Rp (radially outward). |
| 36 | `P1_F_RKx_N` | N | Radial force at the ball joint (Ivantysyn 4.79, with the equivalent centrifugal force), x component. |
| 37 | `P1_F_RKy_N` | N | Radial force at the ball joint, y component. |

### block forces

| # | Column | Unit | Meaning |
|---|---|---|---|
| 38 | `BLK_F_clamp_N` | N | Cylinder block clamping force onto the valve plate (+z): chamber pressure on the bore bottoms + spring. |
| 39 | `BLK_F_Rx_N` | N | Radial force on the block from the 9 pistons, x. |
| 40 | `BLK_F_Ry_N` | N | Radial force on the block from the 9 pistons, y. |
| 41 | `BLK_F_bearing_N` | N | Radial bearing reaction magnitude. |

### block moments

| # | Column | Unit | Meaning |
|---|---|---|---|
| 42 | `BLK_M_VPx_Nm` | N.m | External tilting moment on the block about the valve-plate centre, x (with the bearing reaction). |
| 43 | `BLK_M_VPy_Nm` | N.m | External tilting moment on the block about the valve-plate centre, y (with the bearing reaction). |

### block forces

| # | Column | Unit | Meaning |
|---|---|---|---|
| 44 | `BLK_xB_mm` | mm | Centre of the chamber clamping forces, x (Ivantysyn 4.110). |
| 45 | `BLK_yB_mm` | mm | Centre of the chamber clamping forces, y. |

### swash plate

| # | Column | Unit | Meaning |
|---|---|---|---|
| 46 | `SW_F_normal_N` | N | Total normal force on the swash plate from the 9 pistons (+ = pressing). |
| 47 | `SW_Fx_N` | N | Force on the swash plate, x. |
| 48 | `SW_Fy_N` | N | Force on the swash plate, y. |
| 49 | `SW_Fz_N` | N | Force on the swash plate, z. |
| 50 | `SW_Mx_Nm` | N.m | Moment of the normal forces on the swash plate about the origin, x. |
| 51 | `SW_My_Nm` | N.m | Moment about the origin, y. |
| 52 | `SW_Mz_Nm` | N.m | Moment about the origin, z. |

### slipper film

| # | Column | Unit | Meaning |
|---|---|---|---|
| 53 | `SL1_h0_um` | um | Slipper 1 film thickness at the centre of the land (h0). |
| 54 | `SL1_h0deg_um` | um | Slipper 1 film at the land outer radius, 0 deg (sliding direction xi). |
| 55 | `SL1_h120deg_um` | um | Slipper 1 film at the land outer radius, 120 deg (from xi towards zeta). |
| 56 | `SL1_h240deg_um` | um | Slipper 1 film at the land outer radius, 240 deg. |
| 57 | `SL1_hmin_um` | um | Slipper 1 minimum film (negative = inside the contact penalty). |
| 58 | `SL1_tilt_xi_mrad` | mrad | Slipper 1 tilt about zeta (film slope along xi). |
| 59 | `SL1_tilt_zeta_mrad` | mrad | Slipper 1 tilt about xi (film slope along zeta). |
| 60 | `SL1_p_pocket_bar` | bar | Slipper 1 pocket pressure (absolute). |
| 61 | `SL_mean_h0_um` | um | Mean centre film h0 of the 9 slippers. |

### slipper leakage

| # | Column | Unit | Meaning |
|---|---|---|---|
| 62 | `SL1_leak_LPM` | L/min | Slipper 1 leakage out of chamber 1 (+ out of the chamber). |
| 63 | `SL_mean_leak_LPM` | L/min | Mean leakage per slipper (9 slippers). |
| 64 | `SL_all_leak_LPM` | L/min | Total leakage of the 9 slippers. |

### slipper forces

| # | Column | Unit | Meaning |
|---|---|---|---|
| 65 | `SL1_F_piston_N` | N | Normal force from piston 1 on slipper 1 (+ = pressing on the swash plate). |
| 66 | `SL1_F_fluid_N` | N | Fluid film force on slipper 1 (pocket + land, incl. squeeze), + = lifting off the plate. |
| 67 | `SL1_F_contact_N` | N | Asperity contact force on slipper 1. |
| 68 | `SL1_F_retainer_N` | N | Retainer (hold-down) force on slipper 1. |
| 69 | `SL1_F_friction_N` | N | Viscous friction on the slipper 1 face. |

### slipper moments

| # | Column | Unit | Meaning |
|---|---|---|---|
| 70 | `SL1_M_fluid_xi_Nm` | N.m | Fluid tilting moment on slipper 1 about the ball joint, xi-tilt. |
| 71 | `SL1_M_fluid_zeta_Nm` | N.m | Fluid tilting moment on slipper 1 about the ball joint, zeta-tilt. |
| 72 | `SL1_M_contact_xi_Nm` | N.m | Contact tilting moment on slipper 1, xi-tilt. |
| 73 | `SL1_M_contact_zeta_Nm` | N.m | Contact tilting moment on slipper 1, zeta-tilt. |
| 74 | `SL1_M_ext_xi_Nm` | N.m | External tilting moment (friction + inertia) on slipper 1, xi-tilt. |
| 75 | `SL1_M_ext_zeta_Nm` | N.m | External tilting moment on slipper 1, zeta-tilt. |

### valve-plate film

| # | Column | Unit | Meaning |
|---|---|---|---|
| 76 | `VP_h0_um` | um | Block / valve-plate film thickness at the axis (h0). |
| 77 | `VP_h0deg_um` | um | Valve-plate film at the outer land radius, 0 deg (+x). |
| 78 | `VP_h120deg_um` | um | Valve-plate film at the outer land radius, 120 deg (from +x towards +y). |
| 79 | `VP_h240deg_um` | um | Valve-plate film at the outer land radius, 240 deg. |
| 80 | `VP_hmin_um` | um | Minimum valve-plate film. |

### valve-plate leakage

| # | Column | Unit | Meaning |
|---|---|---|---|
| 81 | `VP_K1_leak_LPM` | L/min | Leakage out of kidney 1 (chamber 1) into the film. |
| 82 | `VP_mean_leak_LPM` | L/min | Mean kidney leakage per chamber (9 kidneys). |
| 83 | `VP_all_leak_LPM` | L/min | Total leakage out of the 9 kidneys into the film. |
| 84 | `VP_case_leak_LPM` | L/min | Valve-plate leakage reaching the case (inner + outer land edges; includes flow from the ports). |

### valve-plate forces

| # | Column | Unit | Meaning |
|---|---|---|---|
| 85 | `VP_F_clamp_N` | N | Clamping load on the film (block external axial load). |
| 86 | `VP_F_fluid_N` | N | Fluid lift force of the film (incl. squeeze). |
| 87 | `VP_F_contact_N` | N | Asperity contact force between block and valve plate. |

### valve-plate moments

| # | Column | Unit | Meaning |
|---|---|---|---|
| 88 | `VP_M_ext_x_Nm` | N.m | External moment on the block about the valve-plate centre, x. |
| 89 | `VP_M_ext_y_Nm` | N.m | External moment on the block about the valve-plate centre, y. |
| 90 | `VP_M_fluid_x_Nm` | N.m | Fluid film moment about the valve-plate centre, x. |
| 91 | `VP_M_fluid_y_Nm` | N.m | Fluid film moment about the valve-plate centre, y. |
| 92 | `VP_M_contact_x_Nm` | N.m | Contact moment about the valve-plate centre, x. |
| 93 | `VP_M_contact_y_Nm` | N.m | Contact moment about the valve-plate centre, y. |
| 94 | `VP_M_bearing_x_Nm` | N.m | Moment carried by the bearing (block kept parallel to the plate), x. |
| 95 | `VP_M_bearing_y_Nm` | N.m | Moment carried by the bearing, y. |

### piston film

| # | Column | Unit | Meaning |
|---|---|---|---|
| 96 | `PB1_e_case_r_um` | um | Piston 1 eccentricity at the bore exit (case end), radial direction r. |
| 97 | `PB1_e_case_t_um` | um | Piston 1 eccentricity at the bore exit, tangential direction t = z x r. |
| 98 | `PB1_e_ch_r_um` | um | Piston 1 eccentricity at the piston end face (chamber end), r. |
| 99 | `PB1_e_ch_t_um` | um | Piston 1 eccentricity at the end face, t. |
| 100 | `PB1_h_case_0deg_um` | um | Piston 1 film at the bore exit (case end), 0 deg from r towards t. |
| 101 | `PB1_h_case_120deg_um` | um | Piston 1 film at the bore exit (case end), 120 deg from r towards t. |
| 102 | `PB1_h_case_240deg_um` | um | Piston 1 film at the bore exit (case end), 240 deg from r towards t. |
| 103 | `PB1_h_ch_0deg_um` | um | Piston 1 film at the piston end face (chamber end), 0 deg from r towards t. |
| 104 | `PB1_h_ch_120deg_um` | um | Piston 1 film at the piston end face (chamber end), 120 deg from r towards t. |
| 105 | `PB1_h_ch_240deg_um` | um | Piston 1 film at the piston end face (chamber end), 240 deg from r towards t. |
| 106 | `PB1_hmin_um` | um | Piston 1 minimum film (radial clearance = half the diametral clearance when centred). |

### piston leakage

| # | Column | Unit | Meaning |
|---|---|---|---|
| 107 | `PB1_leak_LPM` | L/min | Piston 1 / bore leakage out of chamber 1 (+ out; includes the drag of the sliding piston). |
| 108 | `PB_mean_leak_LPM` | L/min | Mean piston / bore leakage per piston (9 pistons). |
| 109 | `PB_all_leak_LPM` | L/min | Total piston / bore leakage of the 9 pistons. |

### piston film forces

| # | Column | Unit | Meaning |
|---|---|---|---|
| 110 | `PB1_F_ext_r_N` | N | External lateral force (swash-plate reaction, slipper friction, centrifugal) on piston 1, radial direction r. |
| 111 | `PB1_F_ext_t_N` | N | External lateral force (swash-plate reaction, slipper friction, centrifugal) on piston 1, tangential direction t. |
| 112 | `PB1_F_fluid_r_N` | N | Fluid film lateral force (incl. squeeze) on piston 1, radial direction r. |
| 113 | `PB1_F_fluid_t_N` | N | Fluid film lateral force (incl. squeeze) on piston 1, tangential direction t. |
| 114 | `PB1_F_contact_r_N` | N | Bore contact lateral force on piston 1, radial direction r. |
| 115 | `PB1_F_contact_t_N` | N | Bore contact lateral force on piston 1, tangential direction t. |
| 116 | `PB1_F_friction_N` | N | Viscous axial friction of the bore film on piston 1 (+z). |

### piston film moments

| # | Column | Unit | Meaning |
|---|---|---|---|
| 117 | `PB1_M_ext_r_Nm` | N.m | Moment of the external forces about the piston 1 end-face centre, from the r-forces (sum F_r * zeta). |
| 118 | `PB1_M_ext_t_Nm` | N.m | Moment of the external forces about the end-face centre, from the t-forces (sum F_t * zeta). |
| 119 | `PB1_M_fluid_r_Nm` | N.m | Moment of the fluid film about the piston 1 end-face centre, from the r-forces (sum F_r * zeta). |
| 120 | `PB1_M_fluid_t_Nm` | N.m | Moment of the fluid film about the end-face centre, from the t-forces (sum F_t * zeta). |
| 121 | `PB1_M_contact_r_Nm` | N.m | Moment of the bore contact about the piston 1 end-face centre, from the r-forces (sum F_r * zeta). |
| 122 | `PB1_M_contact_t_Nm` | N.m | Moment of the bore contact about the end-face centre, from the t-forces (sum F_t * zeta). |

### total leakage

| # | Column | Unit | Meaning |
|---|---|---|---|
| 123 | `LEAK_P1_LPM` | L/min | Total leakage out of chamber 1 (slipper + kidney + piston bore). |
| 124 | `LEAK_all_LPM` | L/min | Total leakage out of the 9 chambers (all gaps). |
