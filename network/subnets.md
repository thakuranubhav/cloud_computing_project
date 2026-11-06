# Task 5 — Subnetting

Address block: `10.20.0.0/16`

Allocate from largest host requirement to smallest.

| Subnet | Hosts needed | Network | Prefix | Usable host range | Broadcast | Usable hosts |
|---|---:|---|---|---|---|---:|
| app | 1000 | 10.20.0.0 | /22 | 10.20.0.1 – 10.20.3.254 | 10.20.3.255 | 1022 |
| public | 250 | 10.20.4.0 | /24 | 10.20.4.1 – 10.20.4.254 | 10.20.4.255 | 254 |
| data | 100 | 10.20.5.0 | /25 | 10.20.5.1 – 10.20.5.126 | 10.20.5.127 | 126 |
| management | 20 | 10.20.5.128 | /27 | 10.20.5.129 – 10.20.5.158 | 10.20.5.159 | 30 |

## Working

- 1000 hosts require at least 1022 usable addresses ? `/22`.
- 250 hosts require at least 254 usable addresses ? `/24`.
- 100 hosts require at least 126 usable addresses ? `/25`.
- 20 hosts require at least 30 usable addresses ? `/27`.

The subnets are placed consecutively starting from `10.20.0.0`, as required.
