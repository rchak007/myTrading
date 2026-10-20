# Beth Kindig — Top 20 AI Stocks for Q4 2026

**Source:** I/O Fund, `io-fund.com/premium/beth-kindigs-top-20-ai-stocks-for-q4-2026`
**Author:** Beth Kindig · **Published:** 2026-10-20 · **Length:** ~196 min read (100+ pages)
**Captured:** 2026-10-19 (web page select-all, reorganised; numbers preserved verbatim)

> This is a **reference capture**, not analysis. Figures below are the author's
> and the companies' — not independently verified by this system. Where the
> article hedges ("roughly", "estimated", "back-of-napkin"), that hedge is kept.

---

## 0. Quick index — tickers covered

| Theme | Tickers |
|---|---|
| Networking | LITE, AAOI, SITM, MXL, MTSI, COHR, ALAB |
| Memory / storage | MU, SK Hynix (ADR), SNDK, SIMO, STX |
| Semis & equipment | AMD, LRCX, TSM*, AVGO* |
| Energy / power | BE, GEV |
| Software | NET, CRWD, PLTR* |

`*` = honorable mention, not a Top 20 slot.

**Also named but unnamed (Discovery/Advanced tiers, teased only):** a CXL stock,
a metrology/process-control stock, a networking stock with 3 catalysts, a power
management stock (~50% AI revenue), an ex-EV burn-in/test equipment stock
(~200% growth), two power producers, a CXL KV-cache server stock.

---

## 1. Performance context (the author's own book)

- Up **+70% YTD** before the recent volatility; **+61%** as of Oct 16.
- Nasdaq-100 **+16%** YTD → lead widened to **~4X** the benchmark, up to 6X
  influencer-led ETFs.
- Raised cash to **25%** in late September ahead of the correction.
- **14 positions** beat the NDX YTD. Five closed for realized gains: AEHR, CLS,
  NBIS, GOOGL, INOD. Nine still held: **AMD, LITE, BE, ALAB, COHR, AAOI, GEV,
  SNDK, MU**.
- 2026 cadence: raised >30% cash, 7 buy alerts in March; back to 30% cash in
  mid-May, 11 buy alerts within a week of the June 26 low.

---

## 2. Thesis 1 — HBM is the #1 constrained component in AI

The central claim of the report. Not just "a bottleneck" — the shortage is now
**dictating product roadmaps at the largest AI companies**.

### Roadmaps bending to supply

- Rubin: 288GB HBM4, same capacity as Blackwell B300's 288GB HBM3e, but
  bandwidth **2.8X** — 8 TB/s → **22 TB/s** (double-width interface + higher speeds).
- Nvidia reportedly testing **8-Hi** HBM4 configs (as low as **192GB**) instead
  of 12-Hi. Fewer dies per stack → same HBM bit output spreads across **more GPUs**.
- Rubin Ultra evaluation widened from 12-Hi HBM4e to include 8-Hi HBM4e,
  12-Hi HBM4, 8-Hi HBM4.

### The counterintuitive pricing outcome

- 8-Hi has **less** DRAM to spread base-die cost over → **higher cost per Gb**.
- TrendForce: 8-Hi carries a **10–20% price premium per Gb** over 12-Hi in 2027.
- Micron pushes back on de-spec fears: lower content per platform "enable[s]
  them to ship more units… bodes well for the industry pricing environment."

### The 2027 shortfall math

| Input | Figure |
|---|---|
| Nvidia Rubin shipments 2027 (UBS) | 8.8M |
| Google TPU shipments 2027 | 8.8–10M |
| Those two combined | ~¾ of total HBM demand |
| + AMD Helios + hyperscaler custom silicon | → **>12% HBM shortfall** |

Caveats the article flags: assumes 100% supply→finished-chip conversion (no
yield loss, not feasible), and **excludes Rubin Ultra**, which would widen it.

### Why conventional DRAM also explodes

- Agentic AI: thousands→millions of concurrent agents need fast access to
  context, tool outputs, KV cache. HBM too scarce/expensive; NAND too slow.
- Context windows: GPT-3.5 Turbo (2023) **4k** → GPT-5.5 xhigh **922k** = **230X in 3 years**.
  GPT-5.6 Sol / GPT-6 Astra ~**1.05M**. Llama 4 Scout **10M** (industry largest).
- A single GB200 NVL72 theoretically supports ~**22 concurrent requests** at 128K
  context on GPT-4. OpenAI has **900M weekly actives** → tens of millions of
  concurrent requests at peak.

### The trade ratio — HBM cannibalises DRAM

- HBM3: **3:1** — 100 bits of HBM costs 300 bits of DRAM supply.
- HBM4/4e: approaching **4:1** → ~33% more DRAM-equivalent capacity per HBM bit.
- Micron: "future more complex HBM having higher trade ratios… also going to be
  constraining supply."

### Fab capacity is going to HBM, not DRAM

| Supplier | 2027 HBM capacity move |
|---|---|
| Samsung | +40% overall to ~250K wpm; HBM4/4e = 80% of that (200K wpm, +190% vs ~70K in 2026) |
| Micron | +150% YoY, 40K wpm (2025) → ~100K wpm by end-2026 |
| SK Hynix | HBM share of DRAM capacity 30% (2026) → **40%** (2027) |

Net: total DRAM wafer capacity can **rise** in 2027 while capacity available for
**conventional** DRAM **declines**.

HBM pricing growth estimates for 2027 range **+44% (low)** to **+121% (high)**.

### CXL as partial relief

- Does **not** solve HBM scarcity — it **offloads** non-bandwidth-critical data
  to conventional DRAM (tiered memory).
- DRAM ≈ **50% of Azure server cost**, ~40% of Meta rack cost. DRAM prices
  projected **+40–45%** next year → a 7% cost reduction saves billions.
- CXL works with legacy/older DDR4 modules → up to **25%** cost saving.
- Microsoft's **Pond** memory pooling: ~**10%** less cluster DRAM demand,
  **7%** lower cost, near-equivalent performance (reduces "memory stranding").

---

## 3. Thesis 2 — Google and AMD attack Nvidia from two different flanks

Nvidia's accelerator share is forecast to fall to **69.7% (bull)** / **59.7%
(base)**, with ASIC servers at 27.8% and AMD at 10% in the base case.

### Google's angle: large, shared, coherent memory

- TPU **8i pod = 1,152 TPUs** (4.5X the prior 256-TPU Ironwood pod).
- Pod-level HBM **331.8TB**, up **7X** from 49.2TB.
- ICI (inter-chip interconnect) gives all 1,152 TPUs a **shared coherent memory pool**.
- Benefits: (1) no duplicated weights/KV cache → each GB of HBM goes further;
  (2) long-context frontier models at minimal latency → inference efficiency;
  (3) contrast Nvidia NVL72 — coherency only across **72 GPUs / 20.7TB**; scaling
  to 1,152 GPUs spans **16 racks** with no unified pool.
- Broadcom supplies Google TPUs **through 2031** (chip-to-chip interconnect, SerDes).
- Investor read-through: **networking improves memory utilisation**, reducing HBM
  needed per workload. Supports large optical/networking positions.

### AMD's angle: maximum local HBM

- **432GB per MI455X** — 12 HBM4 stacks, industry-leading, sitting close to the GPU.
- Higher bandwidth / lower latency than any interconnect-mediated pool.
- More risk (HBM is the scarce thing) but a performance edge: more weights and
  KV cache stay local, less movement across the interconnect.

### Nvidia pursues both

- Rubin: 288GB HBM4, 22 TB/s → agrees with AMD on local capacity.
- NVLink domain roadmap: 72 → 576 → **1,152 GPUs** (same number as Google's pod),
  forcing copper → **co-packaged optics** across multiple racks.

**Takeaway:** on individual hardware specs (coherent memory size, HBM per GPU),
Nvidia's lead is eroding alongside its share. Still the one to beat.

---

## 4. Thesis 3 — 2027 is a supply-chain squeeze unlike anything yet

Three compounding factors:

1. **Complexity per chip rising.** 5.5X reticle CoWoS enables Helios and Rubin.
   TSMC moving to 9X and 14X reticle by end of decade. Failures become
   prohibitively expensive → metrology and test become central.
2. **Competitors piling into fabs** — memory fabs far more constrained than TSMC.
   Relief ~**2028 optimistically, 2030 realistically**.
3. **More chips per system.** NVLink 72→576→1,152; Google 256→1,152; AMD 8→72
   (Helios) and tracking Nvidia after.

### Rubin Ultra redesign (advanced packaging as the limiter)

| Design | Compute dies | I/O | HBM | Packaging |
|---|---|---|---|---|
| Rubin | 2 near-reticle | 2 | 8× HBM4 | CoWoS-L, ~4X reticle interposer |
| Rubin Ultra (old) | 4 near-reticle | 2 | 16× HBM4e | Massive CoWoS-L / stitched |
| Rubin Ultra (new) | **2 reticle-sized** | — | 8× HBM4e | 2 dual-die chips per blade |

Reason for the retreat: at ~**8X reticle size**, severe **substrate warpage** and
yield problems unsolvable in time for the 2027 cycle.

CoWoS supply-demand gap narrows **20% → 10%** by end-2026 (TSMC ramping at **80%
CAGR**), but the constraint then **migrates** CoWoS → SoIC → reticle scaling →
substrates (glass substrates next).

---

## 5. Thesis 4 — Five reasons 2027 is pivotal for networking

Unlike memory (3 suppliers), networking is **fragmented** — dozens of potential
winners shifting almost overnight.

### 5.1 Next-gen systems enter volume production

Networking BOM per system (Bernstein):

| System | Networking content |
|---|---|
| H100 era | baseline |
| GB200 NVL72 | **$929,000** (~7X H100) |
| Vera Rubin | **$1.17M** |
| AMD MI455 Helios | **$1.47M** |
| Google Ironwood | $539,000 |
| OpenAI Jalapeno | $814,000 |

### 5.2 The protocol war is good for investors

Nvidia networking: **$31.4B** in FY26 (Jan) → **$70B run rate** ~3 quarters later.
10X since the 2020 Mellanox deal. Q4 FY26 networking **+3.6X YoY**; April quarter
**+199% YoY / +35% QoQ**; last quarter moderated to **+18% QoQ**.

The fragmentation that helps smaller vendors:

- Nvidia Rubin → **NVLink 6** (plus **NVLink Fusion** to let third-party
  accelerators in)
- AMD Helios → open **UALink** + Ethernet
- Google → own **ICI** fabric + **Boardfly** topology
- Amazon → **NeuronSwitch** (Trainium3)
- Meta / OpenAI → Ethernet with Broadcom
- Anthropic → blend (TPUs + Trainium + Nvidia)

Marvell's point: winners may be those serving **multiple** protocols. Directionally,
the industry-wide effort to escape NVLink lock-in favours smaller players.

### 5.3 1.6T moves into volume; 3.2T next

- TrendForce: >800G transceivers reach ~**100M units** in 2027, roughly doubling.
- Goldman: 800G+1.6T **78M (2026) → 120M (2027)**, +54%; 1.6T alone **+61% to 71M**.
- Optical attach per GPU **doubles**: 3 modules (Blackwell Ultra) → **6** (Rubin).
  Per NVL72 rack: **216 → 432** 1.6T modules.
- At ~60,000 racks: Nvidia alone ≈ **13M modules (2026) → 26M (2027)**, roughly a
  **quarter of the entire >800G market**.
- Rubin introduces **CPO for scale-out** (Spectrum-X Ethernet Photonics), which
  *replaces* some pluggable content.
- Beyond 2027 unit growth decelerates → **growth vector shifts to higher data
  rates** (higher ASP, higher margin).

### 5.4 Scale-up picks up speed

- Lumentum sizes phase one of scale-up at **3–4X** the initial scale-out CPO
  opportunity, **up to 10X** once optics moves inside the rack.
- Goldman scale-up TAM: **$7.2B (2026) → $20.4B (2028)** low case = 3X;
  **$82.8B** high case = **>11X in two years**.
- Optics in scale-up at 576 GPUs: latency **320 ns** vs **>1,500 ns** for a copper
  scale-out equivalent — a ~5X improvement.

### 5.5 NPO moves inside the rack

- CPO scale-up contributes revenue **late 2027 → 2028**; a ~**six-quarter gap**
  where **NPO** finds its footing. Astera: NPO ramps **2027**, CPO **2028**.
- Lumentum CEO on hyperscaler motive: *"in order for us to leapfrog… Nvidia,
  we're going to have to move quickly and have more 1.6T lanes going down the
  back plane… you're going to have to use a lot more optical engines. So the
  size of it looks like it's frankly bigger than even the CPO opportunity."*
- Lumentum now plots NPO TAM potentially exceeding its **$15B CPO TAM**.

### Honourable mentions

- **CPU-centric:** Rubin doubles NICs per compute tray to **8**. As CPU:GPU moves
  toward 1:2 then 1:1, PCIe switch + NIC demand could grow **~10X by end-2027**
  to as much as **52.2M units**.
- **Scale-across:** more a 2H 2027 / 2028+ story; multi-rail promises **32X**
  density for multi-region connectivity.

---

## 6. The CPU story — agentic AI rewrites it

- CPUs account for **50–90% of total latency** in workflows → CPU:GPU ratio must rise.
- AMD server CPU TAM: **$60B → $120B** by 2030 (CAGR 18% → 35%), then raised again
  to **$220B** by 2030 two months later at Advancing AI.
- Historically single-digit growth. Server CPU **units**: +9% (2025) → **+31%
  (2026)** → **+43% (2027)**.
- Mizuho: $209B TAM by 2030; agentic share **4% (2026) → 30% (2030)**, a **123%
  CAGR** to ~$80B.
- Arm CEO: CPU cores per GW **30M → 120M (4X)**; later "4X, 8X, 10X, it's a hard
  number to predict"; back to 4X by late-July earnings.
- Current CPU:GPU in AI DCs **1:4 to 1:8**. Agentic apps (e.g. Meta's Muse) →
  closer to **1:1**, with rumours of **4:1 or higher** for some workloads.
- Shortages already: ~10–15% higher ASPs, Intel backlog >6 months, AMD lead times
  8–10 weeks.

---

## 7. AI servers — half of AI spend from the ACIE segment

**ACIE** = AI Clouds, Industrial and Enterprise (i.e. *non*-hyperscaler).

- Nvidia FQ2: hyperscalers **$48.7B** (+101.2%) vs ACIE **$40.3B** (**+138.5%**) —
  ACIE growing faster.
- Dell: **200 GW** of AI DC growth through 2030, AI = 75% of all DC demand;
  **half** (100 GW) from neoclouds/sovereigns/enterprise → **$1T TAM** over 4 years.
- Why: these customers lack hyperscaler budgets for custom silicon; they want
  **ready-to-deploy** servers online fast — Dell/SMCI can do **6 hours to 1 day**.
- OEM value-add rises with rack complexity: services, monitoring software, higher
  attach of storage and networking. Arm 30-blade design "setting records for air
  cooled"; a 200kW fully liquid-cooled open rack with Super Micro packs **168
  blades / 336 CPUs / up to 45,696 cores**.
- Liquid cooling reaches **53% share** in 2026.

---

## 8. AI energy — power *distribution* (800 VDC)

### The density wall

- Rack power: Blackwell GB200 **120kW** → Blackwell Ultra **180kW** (+50%) →
  Rubin **300–600kW** (a **5X** increase 2025→2027). Excludes networking,
  interconnect, cooling.
- ~**70%** of data centers were built for **4–9kW** racks; **<2%** can handle
  even 50kW.
- Nvidia: at ~200kW, 54 VDC distribution hits its **physical limit**; repeated
  AC/DC conversions are inefficient and add failure points.

### The 800 VDC answer

- At **1 MW**: 54V needs ~**18,500 amps** vs **1,250 amps** at 800V — a ~**15X**
  current reduction, so far less conductor and far lower resistive loss.
- Convert to 800 VDC at the **data center perimeter**, not inside the rack.
- Nvidia claims: **30% lower TCO**; row-level 800 VDC busways carry **85% more
  power** through the same conductor while cutting copper **45%** vs 415 VAC;
  **up to 5%** better end-to-end efficiency; **up to 70% lower** maintenance cost.
- Density: at 54 VDC a megawatt-scale **Kyber** rack would need up to **64U of
  power shelves** (112 inches) — "no room for compute."

### Rack roadmap

| Platform | GPUs/rack | Racks |
|---|---|---|
| NVL72 | 72 | 1 |
| NVL144 (Kyber) | 144 | 1 |
| NVL576 | 72 | 8 |
| NVL1152 (Kyber) | 144 | 8 |

### The content opportunity (GaN / SiC)

- Onsemi: today's 120kW racks ≈ **$9,500–$15,000** content → 800V systems
  ≈ **$115,000** (~10X). Renesas projects **>10X** by end of decade at 1MW racks.
- Mix shifts: today ~**70/30** medium-low vs high voltage → ~**50/50** at 800V.
  High-voltage content **>12X** ($4,500 → ~$57,500); medium/low **5–6X**
  ($10,500 → ~$57,500).
- Timing: 800V is a **2027–2028 story** (Monolithic Power, Onsemi).

---

## 9. AI energy — power *generation*

### Shortfall estimates by 2030 (wide divergence)

| Source | US shortfall |
|---|---|
| McKinsey | 30–55 GW (VA 16.6, TX 10.0, OH 7.4) |
| Morgan Stanley (2029) | 33–59 GW |
| BCG | 80 GW |
| Bank of America | >100 GW |
| TrendForce | **>170 GW** |

- NERC: US summer **peak** demand +**224 GW** over 10 years — a **69% increase**
  over last year's 132 GW projection.

### ERCOT's queue is not what it looks like

- Data center interconnection requests: **355.8 GW** (Apr 2026, ~87% of large-load)
  → **~392 GW** by mid-June (~90%), **+36.2 GW in two months**.
- That is **>9X** the 43.6 GW of AI data centers planned/under construction/
  operating across the entire US.
- Vistra: **>70%** haven't submitted studies; realistic queue is "**12 to 15
  gigawatts by 2030**" — i.e. the headline may overstate by up to **~25X (97%)**.

### 2027 regulatory outlook

- Texas wholesale ~**$30/MWh** vs Lazard's **$51–$129/MWh** levelized cost for a
  new combined-cycle gas plant → **existing generation is increasingly attractive**.
- PJM proposing **BYONG** (bring your own new generation) for firm grid access,
  and clarifying **colocation** rules — favouring owners of large existing fleets.
- Some behind-the-meter generation may revert to **backup/supplemental** as
  utility capacity arrives 2H 2027 → 2028.

### Behind-the-meter still stands

Time-to-power thesis (first raised June 2024; Bloom identified on Discovery
March 2025): *"it is not enough to say the AI economy needs more power, but
rather it needs power **urgently**."* Grid/nuclear can serve the first; not the
second. **Bloom Energy named the author's #1 stock of 2026.**

---

## 10. The Top 20 — per-stock detail

### 10.1 Networking

---

#### LITE — Lumentum ("the networking darling")

**Positioning:** EMLs → CW lasers → OCS → NPO/CPO. Present at every transition.

- 200G/lane now **25%** of EML revenue; majority of shipments expected in 2027.
- **>50% EML unit growth** by Dec-2026 quarter YoY, *even while* allocating
  capacity to CW lasers.
- CW lasers: more competition, no modulator → lower ASP/margin, but "commanding a
  significant price premium… the margin opportunity is still better on EMLs. But
  that gap has closed considerably."
- **OCS doubled QoQ**, expected **$100M** in fiscal Q1. Pump lasers **+80% YoY**
  and **sold out**. Narrow linewidth lasers **+130% YoY**.
- UHP laser demand ramps **2H 2027**; **ELS module** ships around then → moves
  from laser chips to higher-ASP modules.
- Quote on scale: *"for one major hyperscaler, the network capacity connecting
  just two AI data center sites is double the total global backbone capacity they
  built over the entirety of the last decade."*

| Metric | Q4 | Guide / note |
|---|---|---|
| Revenue | **+109% YoY, +24.5% QoQ** | 3rd straight 20%+ QoQ; Q1 guide **+134.2% YoY** |
| Components | $649.4M (64.5% of rev) | +102.7% YoY, +21.8% QoQ |
| Systems | $356.9M (35.5%) | +122.5% YoY, +29.7% QoQ; FY +90.7% |
| GAAP EPS | $(84.65) | debt extinguishment, see below |
| Adj EPS | **$3.23** (+267% YoY) | beat $2.97 by 8.8%; Q1 guide **$4.20** (+281.8%) |
| FY27 EPS | GAAP $16.04 / Adj **$18.73** (+116%) | |
| GAAP GM | 47.4% (+3.2 QoQ, +14.1 YoY) | |
| Adj GM | **50.4%** (+2.5 QoQ, +12.6 YoY) | above the $5B model's 45–48% |
| GAAP OM | 27.8% (+6.2 QoQ, +29.5 YoY) | |
| Adj OM | 36.6% (+4.4 QoQ, +21.6 YoY) | Q1 guide **40.0%** |
| Adj net margin | 32.4% | GAAP net margin (711.7%) on a **$7.76B non-cash** loss on converts |
| Cash / debt | $2.74B / $1.64B | converts equitized — debt roughly halved QoQ |
| AR / Inv | $520.3M (+18% QoQ) / $691.6M (+9.3%) | |

**Valuation:** 12X fwd sales (TTM 25X, 3yr avg 7X); **40X fwd earnings**
(TTM 158X). vs SITM 15X/45X, AAOI 6X/117X, COHR 4X/26X.
Hit the **$5B annualized run rate a quarter early**.

**Risks:** CW laser competition and lower margins; constrained InP; any NPO/CPO
(or system) delay pushes out the 2027 catalyst.

---

#### AAOI — Applied Optoelectronics

**The structural point:** CW + silicon photonics needs **less InP** than EML.

| Architecture | InP content |
|---|---|
| EML | InP DFB laser **+ InP EAM/modulator** |
| CW + SiPh | InP CW/DFB laser **+ silicon modulator** |

Where four EMLs were once needed, integrated SiPh may need only **two** lower-cost
CW lasers.

- **Capacity:** 200K units/month → **650K/month by year-end**.
- **Vertically integrated:** makes its own InP lasers and production equipment —
  lasers otherwise take **21–24 months** to reach high volume. InP substrate
  supply **secured through end-2027**; moving to 4-inch substrates for 2028–29.
- A China ban would be a **tailwind**: *"customers more aggressively will give us
  much more share, especially for U.S. manufacturers."*

| Metric | Q2 | Guide / note |
|---|---|---|
| Revenue | **$191.9M** (+86.4% YoY, +27.0% QoQ) | 5th straight record; est. $190.5M |
| Q3 guide | **~+42% QoQ**, nearly all 800G | |
| Q4 | mgmt says **>$500M** | vs analyst $410M; FY26 $1.1B stands |
| | at low end $484.5M → **+78% QoQ, +260% YoY** | |
| Data Center | $107.7M | +140.4% YoY, +32.3% QoQ; Q3 napkin ≈ $163.5M (+52% QoQ, +272% YoY) |
| Adj EPS | **$0.06** (est $0.02) | from $(0.07) Q1, $(0.16) YoY; Q3 guide $0.11–0.26 (mid $0.185, **below** ~$0.28 street) |
| GAAP EPS | $(0.28) | widened from $(0.19) |
| GAAP GM | 27.7% (−2.6 YoY, −1.4 QoQ) | **margins are the soft spot** |
| Adj GM | 29.9% (−0.6 YoY, +0.7 QoQ) | Q3 guide ~29.8% — ASP tailwinds not yet in |
| GAAP OM | (12.9%) | opex +37% QoQ |
| Adj net margin | **+2.9%** | first positive since Q4 2023 |
| OCF | $11.6M (6%) | from (56.5%) margin last quarter |
| FCF | **$(265.3M)** ex-prepay, (138.2%) margin | capex ~$227M; total Q2 capital investment **$565.5M incl. $280M equipment prepayments** |
| Cash / debt | $508.8M / $220.3M | **$538.8M raised via ATM since May** |
| Inventory | $278.8M (+35.2% QoQ) | |

**Valuation:** 6X fwd sales (TTM 10X, 3yr avg 5X); **117X fwd earnings** —
far above LITE 40X and COHR 26X, reflecting early-stage profitability.

**Risks:** heavy execution risk on the capacity ramp; margins under pressure;
deeply negative FCF; further equity/debt → dilution or financial risk.

---

#### SITM — SiTime ("timing moves into the heart of the system")

From discrete components around the system → **chiplets, advanced substrates,
integrated modules**.

- Content per rack: *"multiple hundreds of dollars in a fully integrated rack"*
  for training; inference is **2–4X** → plausibly **>$1,000** per integrated
  inference rack.
- Two drivers named: (1) inference infrastructure — XPUs, switches — growing in
  **units and ASPs**; (2) networking bandwidth, especially **1.6T optical modules**
  growing faster than anticipated.
- Transition could add **$2.5B to SAM by 2030** (vs CED's $101.2M this quarter).
- Management's own long-run target: **~30% growth over 5–8 years** — which the
  author flags as conservative given current rates.

| Metric | Q2 | Guide / note |
|---|---|---|
| Revenue | **$157.4M** (+126.5% YoY, +38.6% QoQ) | beat $146.4M by 7.5%; first triple-digit quarter in 5 years |
| Q3 guide | $285–295M → **+84.2% QoQ, +247.1% YoY** | incl. **~$85M inorganic** (Timing Business) |
| Q3 organic | $200–210M → **+30.2% QoQ, ~+145% YoY** | |
| CED segment | **$101.2M** (+33.7% QoQ, +181.1% YoY) | 64% of revenue (was ~52%); ~7X a $60M run rate two years ago |
| GAAP EPS | $0.66 | from $(0.20) Q1; beat $0.58 guide |
| Adj EPS | **$2.34** (+397.2% YoY) | beat $1.95 guide by 20% |
| GAAP GM | 63.0% (+4.0 QoQ, +11.1 YoY) | |
| Adj GM | **67.1%** | beat 65% guide; Q3 guide **68%** |
| GAAP OM | +5.2% (+16.1 QoQ, +40.6 YoY) | |
| Adj OM | **34.0%** | beat 30% guide; Q3 guide **39.6%** |
| GAAP net margin | **11.5%** | double digits |
| Adj net margin | 41.7% | |
| OCF / FCF | $40.0M (25.4%) / $27.1M (17.2%) | 4th straight positive FCF quarter |
| Cash / debt | **$1.92B** / $1.32B | **+$1.3B converts** for the Timing acquisition |
| AR / Inv | $88.7M (+61.3% QoQ) / $103.9M (+14.0%) | |

**Valuation:** **15X fwd sales** — *below* its 3yr avg of 22X and TTM 29X, despite
127% growth. 45X fwd earnings. Premium to LITE 12X, AAOI 6X, COHR 4X.

**Risks:** integration + $1.3B convertible debt + dilution; management's own ~30%
long-term target implies today's triple-digit rates normalise.

---

#### MXL — MaxLinear (optical DSPs)

- **Keystone** (100G/lane, 5nm CMOS PAM4 DSP + SerDes) — twelve PAM4 DSPs across
  400G/800G, bare die, driverless variants, AOC and AEC targets. Pairs with
  **Topanga** and **Washington** TIAs.
- Claimed **~40% lower power** than competing 100G PAM4 DSPs; 7W 400G and 13W
  800G designs.
- **Rushmore** (1.6T) sampling now, **ramps 2027** — layering *on top of* Keystone,
  same customers.
- One of **3 major players** in optical DSP; differentiates on performance, power
  and a **more diversified supply chain**.

| Metric | Q2 | Guide / note |
|---|---|---|
| Revenue | **$168.8M** (+55.2% YoY, +23.1% QoQ) | above the $150–165M range; Q1 was +0.6% QoQ |
| Q3 guide | $210–220M (mid **$215M**) → **+27.3% QoQ, +70.0% YoY** | **six quarters ahead** of consensus (street had this in Q4 2027) |
| FY26 implied | ~$776M (+66%) if Q4 adds another ~$40M | vs $657M estimate pre-print |
| Infrastructure | **$85.0M** (+35.3% QoQ, **+145% YoY**) | first time a **majority** of revenue at 50.4% (was 31.9% YoY) |
| Q3 Infra est. | ~$116.1M (+37% QoQ, **+188% YoY**) at ~54% mix | |
| Keystone FY26 | raised to **$190–210M** (from $150–170M) | vs ~$60–70M in 2025 → **+208% YoY** |
| GAAP EPS | **$0.02** | first positive GAAP quarter after **12 straight losses**; aided by an **$8.3M tax benefit** |
| Adj EPS | $0.35 (+1,650% YoY) | 4th straight quarter of accelerating growth |
| GAAP GM | 57.8% (+1.3 YoY) | Q3 guide 58.5% |
| Non-GAAP GM | 59.5% (flat QoQ) | Q3 guide 60% |
| GAAP OM | (2.5%) (+10.0 QoQ, +20.1 YoY) | Q3 guided to **low double-digit positive** on **flat opex ($101M)** |
| Adj OM | **22.3%** (+6.4 QoQ, +15.1 YoY) | highest since early 2023; opex +6% vs revenue +23% |
| GAAP↔Adj gap | ~24 points | **~$36.5M SBC** / performance equity |
| OCF / FCF | $4.8M (2.8%) / $2.5M (1.5%) | both **down YoY** despite growth — working capital |
| Inventory | $105.5M (+22.9% QoQ) | wafer prepayments |
| Cash / debt | $93.7M / $123.9M LT | revolver extended to Mar 2028, +$30M, **undrawn** |

**Valuation:** 10X fwd sales, **46X fwd earnings**. Below SITM (15X) and MTSI (12X)
on sales; slightly higher on earnings than LITE.

**Risks:** established DSP competitors; Rushmore/hyperscaler delays hit 2027;
**thin FCF**; large SBC-driven GAAP/adjusted gap.

---

#### MTSI — MACOM (deeper in the stack than AAOI or LITE)

Supplies the **components inside** transceivers — TIAs (currently highly
constrained), drivers, lasers, photodetectors.

- **200G PDs:** *"one of the industry's best"*; a **400G PD** already with ~a dozen
  customers. Ramping in volume, a meaningful Data Center contributor.
- **25G DFB lasers:** industry-wide shortage — *"customers are coming to us with
  urgency."*
- **NPO/CPO:** sampling **200G and 400G per-lane TIAs and drivers**; may not move
  revenue until **2028**.
- **75mW CW lasers:** production **2H FY27**; management calls CW a possible
  *"watershed moment"* in 2028.

| Metric | FQ3 | Guide / note |
|---|---|---|
| Revenue | **$342.2M** (+35.7% YoY, +18.4% QoQ) | est. $335.6M; fastest sequential in **9+ years** |
| FQ4 guide | $415–425M (mid **$420M**) → **+22.7% QoQ, +60.8% YoY** | ~15% above the $365.6M estimate |
| Data Center | **$137.6M** (+81.5% YoY, +40.1% QoQ) | beat the 35% QoQ guide |
| FQ4 DC guide | +35% QoQ again ≈ **$185.7M** → **+133.3% YoY** | FY DC guide raised to **+74%** (from 60%) |
| GAAP EPS | $1.28 (+167% YoY) | includes a **$41.5M gain** on its IQE plc stake |
| Adj EPS | $1.40 (+55.6%) | 3.7% beat; FQ4 guide **$1.97–2.03** vs $1.57 est → **+112.8% YoY** |
| GAAP GM | 58.3% (+3.0 YoY, +1.4 QoQ) | |
| Adj GM | 59.7% | FQ4 guide **60–61%** |
| GAAP OM | **22.5% (record)** (+7.5 YoY) | |
| Adj OM | **31.5%** (+6.3 YoY) | |
| GAAP net margin | 29.4% | inflated by the IQE gain |
| Adj net margin | 32.1% (+5.0 YoY) | |
| OCF / FCF | $80.0M (23.4%) / $59.2M (17.3%) | both down QoQ from 27.2%/22.7% |
| Cash+STI / debt | $663.0M / $340.5M | |
| AR / Inv | $179.1M (+12% QoQ) / $281.5M (+12% QoQ, +31% YoY) | a **measured** build |

**Valuation:** 12X fwd sales — in line with LITE, below SITM 15X, premium to
COHR 4X and AAOI 6X.

**Risks:** NPO/CPO or qualification delays push revenue to later periods (2028
timeline named); rising hyperscaler dependence; competition in TIAs/drivers/lasers.

---

#### COHR — Coherent (targeting $3B/quarter)

- First **$2B quarter**. Management targets **$3B/quarter by YE27** vs consensus
  **$2.63B**.
- **6-inch InP** relieves a critical constraint — already producing EMLs, CW
  lasers and photodiodes with **better yields than 3-inch**. InP capacity doubles
  by YE2026 and **doubles again** by YE2027.
- CPO seeing **pull-forward**; NPO a **new leg up** over the past 3–6 months.
  **Every major strategic customer** engaged on CPO, NPO or both. Launched
  **Photon Link** integrated-optics platform in September.

Roadmap / incremental SAM (**$20B** on top of a **$50B** current SAM):

| Item | Timing | Note |
|---|---|---|
| OCS | ramping now | TAM **doubled to $4B** |
| CPO / NPO | 1H 2027 | |
| Multi-rail | 1H 2027 | |
| Thermal Solutions | 2H 2027 | |

| Metric | FQ4 | Guide / note |
|---|---|---|
| Revenue | **$2.05B** (+33.7% YoY, +13.3% QoQ) | beat by 3.3%; **+42% pro forma**; first double-digit QoQ since Sep 2022 |
| FQ1 guide | $2.2–2.4B (mid $2.3B) → **+45% YoY, +12.4% QoQ** | 8.1% above $2.13B est |
| DC & Comms | **$1.62B** (+58.6% YoY, +18.6% QoQ) | ~80% of revenue |
| Data Center | **+66% YoY, +24% QoQ** | from +37%/+13% in FQ3 |
| GAAP EPS | $1.19 | from $(0.83) |
| Adj EPS | **$1.74** (+74% YoY) | 7.4% beat; FQ1 guide $1.85–2.05 (+68%), 10.2% above est |
| GAAP GM | 38.5% (+2.8 YoY) | **vs LITE's 47.4%** |
| Adj GM | 40.2% (+2.1 YoY) | FQ1 guide 40.5% — **vs LITE's 50.4%** |
| GAAP OM | 12.4% | **vs LITE's 27.8%** |
| Adj OM | 21.8% (+3.8 YoY) | FQ1 guide 22.7% — **vs LITE's 36.6%** |
| Adj net margin | 17.2% | |
| OCF | $69M (3.4%) | FY26 **$79.5M, 1.1% margin** (was 10.9% in FY25) |
| FCF | **$(486.3M)**, (23.8%) | 3rd straight negative quarter; FY26 **$(1.02B)**, (14.4%) |
| Capex | FY26 **$1.1B** (from $441M) | will stay elevated |
| Cash / debt | $1.16B (from $1.59B) / **$3.22B** | |
| Inventory | **$2.58B** (+21% QoQ) | outpacing revenue growth |

**Valuation:** **4X fwd sales** — the cheapest in the optical group — and **26X fwd
earnings** vs LITE 40X / SITM 45X. Lower margins explain the discount.

**Risks:** the capacity plan is capital-hungry; **$1B negative FCF**; $3.2B debt;
margins well below Lumentum's — margin expansion *is* the thesis.

---

#### ALAB — Astera Labs (Scorpio takes over)

- **Scorpio X-Series 320-lane** is the largest open memory-semantic fabric switch
  on the market: **5.12 TB/s** bidirectional in a single ASIC, 16 lanes per device,
  **20 accelerators per switch** — *"2x the radix in a single hop."*
  One switch for 320 GPUs → fewer hops → lower latency.
- Becomes the **largest product family in Q3** — a quarter early (was Q4).
- Scale-up switch content **well above $1,000 per XPU**. In a 20-XPU config at 16
  lanes → ~**$20,000 of switching content** per switch domain.
- Merchant scale-up switch TAM: **$20B**.
- **Leo CXL memory controllers** — higher volumes across two hyperscalers;
  extends the story into **memory pooling**, timely given DRAM cost.

| Metric | Q2 | Guide / note |
|---|---|---|
| Revenue | **$392.4M** (+104.5% YoY, +27.3% QoQ) | beat $360.8M by 8.8%; QoQ accelerated 13.3 pts |
| Q3 guide | $540–560M → **+40.2% QoQ, +138.5% YoY** | **31.9% above** the $417M consensus — one of the biggest beat/raises of the quarter |
| GAAP EPS | $0.83 (+188.6%) | beat $0.45 by 84.4% — **inflated by a tax benefit** |
| Adj EPS | **$0.80** (+82.6%) | cleaner read; beat $0.69 by 15.9% |
| GAAP GM | 73.3% (**−3.0 QoQ**, −2.6 YoY) | Q3 guide **72%** → 2nd straight compression |
| GAAP OM | 22.7% (+2.6 QoQ) | Q3 guide **29.5%** (+6.8 QoQ) |
| Adj OM | 39.1% (+2.9 QoQ, ~flat YoY) | Q3 guide **43.3%** |
| OCF / FCF | $87.7M (22.3%) / $67.2M (17.1%) | **down 48–52 pts YoY** (Q2'25 had favourable timing at 70.5%) |
| Cash / debt | $1.25B / **zero** | |
| AR / Inv | $192.5M (+42.8% QoQ) / $113.8M (**+89.1% QoQ**) | sizable build ahead of the Q3 ramp |

**Valuation:** **27X fwd sales, 73X fwd earnings** — the most expensive in the
networking group (vs LITE 12X/40X, SITM 15X/45X).

**Risks:** merchant scale-up switching draws larger suppliers and hyperscaler
in-house efforts; thesis increasingly tied to **one product's ramp**; **second
straight quarter of gross margin contraction**; inventory nearly doubled.

---

### 10.2 Memory

---

#### MU — Micron ("revenue growth has peaked but earnings can carry the stock")

The bull/bear fight is about **the cycle top**, not the numbers.

- **Bear case the author partly accepts:** Strategic Customer Agreements (SCAs)
  will likely create a **digestion period** on revenue growth.
- **Bull case:** units keep expanding — server units **high-teens %** growth in
  both 2026 and 2027; DRAM industry bits **low-20s %** in 2027 and 2028; NAND bits
  **mid-20s %**. *"Vast majority"* of CY2027 HBM supply already contracted with
  **"significant price increases year-over-year."**
- Management, repeatedly: **2027 and 2028 much tighter than 2026.**
- Author on Fox Business: **99% probability memory has not peaked**, on agentic
  concurrency alone.

| Metric | Q4 | Guide / note |
|---|---|---|
| Revenue | **$54.23B** (+379.3% YoY, +30.8% QoQ) | QoQ **decelerated ~43 pts** from +73.7% |
| Q1 guide | mid **$61.5B** → +351% YoY, **+13.4% QoQ** | a **60-point** QoQ deceleration in two quarters |
| Street FY27 | **$274.6B (+106%)** | decelerating from FY26's +256%; exit rate +40% YoY / +5.8% QoQ |
| DRAM | **$39.8B** (+27% QoQ, +343% YoY) | ASPs **high-teens % QoQ**, down from **low-60s** in FQ3 |
| NAND | **$14.1B** (+42% QoQ, +526% YoY) | ASPs ~**+30% QoQ**, down from mid-80s |
| FY26 | DRAM **$100.7B** (+252%), NAND $31.8B (+274%) | DRAM >$100B for the first time |
| **CDBU** (Core Data Center) | **$18.0B** (+56% QoQ, **+1,042% YoY**) | now the **largest BU**, 33.2% of revenue (13.9% YoY) |
| CMBU (Cloud Memory) | $16.3B (+18% QoQ, +258% YoY) | decelerated from +78% QoQ |
| MCBU (Mobile/Client) | $13.1B (+14% QoQ, +249% YoY) | price up, bits down |
| AEBU (Auto/Embedded) | $6.8B (+47% QoQ, +376% YoY) | |
| Adj EPS | **$33.42** (+1,003% YoY) | 5% beat; Q1 guide **$38.15** (+698%) |
| FY27 adj EPS | **$176.65 (+134%)** | each $1B revenue beat ≈ **+$0.60–0.65** |
| FY28 adj EPS | **$203.57** (+15%) | |
| GAAP GM | **~87%** | Q1 guided ~86% and called **the floor for FY27** — higher every quarter after |
| Operating margin | Q1 guide **82.2% (record)** | ~$8.20 of sequential revenue per $1 of opex |
| OCF | **~$44B (81% margin)** | FY26 **$89.7B (67%)**, >5X YoY |
| FCF | $33.2B (61%) | FY26 **$62.3B (47%)**; net capex $10.8B (+52% QoQ) |
| Cash / debt | **$73.5B / $5.2B** | incl. **$12.7B of SCA deposits**; a year ago: $11.9B vs $14.6B |

**Valuation:** ~4X fwd sales (TTM 9X); **5X fwd earnings** (TTM ~10X) —
**5.2X FY28**. A re-rate to merely 7X or 9X forward would be significant upside.
FY27 EPS estimate (~$177) is **nearly equal to the share price a year ago ($182)**.

**Risks:** cyclical peak in DRAM/NAND pricing (ASP growth already moderating
sharply); SCAs cap upside and create digestion; industry-wide capacity expansion.

---

#### SK Hynix (ADR) — the HBM leader

~**50% HBM market share**; reportedly secured **70%** of Nvidia's HBM orders for
Vera Rubin.

- Counter to the capacity fear: *"should supply constraints ease… the market's
  growth trajectory could **expand further** as latent demand is met. On the
  supply side, however, it appears difficult for the supply-demand balance to
  improve meaningfully in the near term."*
- **HBM4 mass production began Q2**, "full-fledged ramp-up" in 2H. **HBM4e
  sampling** with a major customer. 2027 supply and pricing under discussion.
- **LTAs with 10 customers**, averaging ~**5 years**. Reportedly **removed the
  price cap** → smoother path to capture upside. Share of sales kept at an
  "appropriate level" to preserve upside.
- DRAM ASPs **+~30%** in Q2. Enterprise SSD revenue **2X QoQ**; Solidigm **30TB+**
  drives **>3X QoQ**. NAND ASPs **+mid-50%**.

| Metric | Q2 | Guide / note |
|---|---|---|
| Revenue | **KRW 79.3T ≈ $54.6B** (+257% YoY, +51% QoQ) | record |
| Q3 est. | ~$75B (+37% QoQ, +339% YoY) | DRAM bits **+~10% QoQ**, NAND bits +low-single |
| DRAM | ~$39.9B, **73% of revenue** (+40% QoQ, +232% YoY) | server ≈ half the segment |
| NAND | ~$14.7B, ~27% (+90% QoQ, +365% YoY) | +6 pts of mix |
| EPS / ADR | ~**$9.06** (Q2) | Q3 ~$6.12, Q4 ~$7.09; **2026 $26.31 → 2027 $32.73 (+24%)** |
| Gross margin | **83%** (+29 YoY, +4 QoQ) | |
| Operating margin | **76% (all-time record)** (+35 YoY) | |
| Net margin | 118% | **Kioxia stake gain**; Q1 was 77% |
| OCF | KRW 65.7T ≈ **$43.8B (83%)** | from 50% in Q1 |
| FCF | KRW 55.0T ≈ **$36.7B (69%)** | capex +39% QoQ to KRW 10.7T (~$7.1B) |
| FY26 capex | high-KRW-40T (~**$31–33B**) | pulling in M15X; Yongin Fab 1 cleanroom early 2027 |
| Cash+STI / debt | KRW 88.0T ≈ $56.8B / KRW 18.6T ≈ $12.0B | excludes July ADR listing proceeds |

**Valuation:** ~4X fwd sales (TTM 8X); **5X fwd earnings** — in line with MU.

**Risks:** HBM4 competition from Micron and Samsung; LTAs cap upside; oversupply
if AI demand moderates; **US export restrictions on its China fabs**.

---

#### SNDK — SanDisk ($94B contracted challenges the cycle)

The thesis is that **contracted revenue breaks the normal memory cycle**.

- *"AI is fundamentally a memory centric, storage intensive problem… Every AI
  interaction creates content that must be stored, retrieved, and served at low
  latency."*
- **KV cache offload:** as caches grow, there's an opportunity to move the
  workload **out of expensive HBM into NAND**.
- **High-bandwidth flash (HBF)** addresses the "memory wall" by bringing
  high-capacity flash closer to compute.

**New Business Model (NBM) agreements** — five announced last April, **five more
this quarter**:

| Item | Figure |
|---|---|
| Total minimum revenue (floor pricing) | **$93.9B** |
| Weighted average duration | **>4 years** (up to 5) |
| RPO at quarter end | $59.8B → **$91.1B** incl. two signed post-close |
| Financial guarantees (deposits etc.) | **$16.5B** |
| Pricing | **variable, not fixed** — not price-capped like a traditional LTA |

| Metric | FQ4 | Guide / note |
|---|---|---|
| Revenue | **$8.97B** (+371.6% YoY, +50.7% QoQ) | ~$1B above the $8B guide; beat est by 6.9%. **⅓ volume, ⅔ price** |
| FQ1 guide | $10.30–10.80B → **+17.7% QoQ** | **below** the $10.82B consensus; sequential dollar growth $3B → $1.6B |
| Data Center | **$2.98B** (+102.9% QoQ, **~+1,298% YoY**) | >33% of revenue; was $269M in Q1 → **11X in a year** |
| FY26 Data Center | $5.15B (+437%) | 25.4% of revenue |
| GAAP EPS | **$43.97** (+90.9% QoQ) | |
| Adj EPS | **$39.25** | vs $30–33 guide |
| FY26 | GAAP $73.76 / Adj $70.88 (+2,271%) | |
| FY27 est | GAAP **$225.94 (+206%)** / Adj **$212.95 (+200%)** | |
| GAAP GM | **84.6%** (+58.4 YoY, +6.2 QoQ) | FQ1 guide ~84% — **approaching a ceiling** |
| GAAP OM | **78.5%** (+77.6 YoY) | started FY26 at **7.6%**; FQ1 guide 78.3% |
| GAAP net margin | 77.0% | adj 68.7%; gap = **$804M mark-to-market** gain |
| OCF | **$7.13B (79.5%)** | incl. **$1.94B NBM deposits/prepayments** (vs $538M in Q3) |
| FCF | $7.08B (79.0%) | **ex-NBM: $5.04B, 56.2%** — the honest run-rate |
| Cash / debt | $4.76B / **zero** | fully repaid earlier in the year |
| AR / Inv | $4.71B (**+72.7% QoQ**, outpacing revenue) / $2.70B (+20.6%) | inventory days held high on purpose for NBM |

**Valuation:** 4X fwd sales (TTM 11X); **6X fwd earnings** (TTM 19X).

**Risks:** ~⅔ of growth was **price**; margins at 85%/80% leave little room — the
risk is **earnings normalisation**, not a volume collapse.

---

#### SIMO — Silicon Motion (boot drives = "crown jewel")

- Boot drive wins now include **Google TPUs** alongside **Nvidia DPUs**, telecom
  and server makers. *"We do have a pretty large share for BlueField supply."*
- **30–40+ boot drives per server rack**, and capacity per drive rising → units
  *and* content scale together.
- Every accelerator class needs one: *"DPU… TPU, NPU… PCIe switch… NVLink switch…
  Ethernet switch, there is a boot drive they need."*
- **DRAM-less architecture** with hardware-level security is the moat — very few
  competitors; SIMO leads. Combines controller + firmware + security.
- Boot drives carry a **higher ASP** because SIMO sells **controller *and* NAND**.
  At 256GB density, NAND makers don't compete there; neither do module makers.
- **MonTitan:** the 2027 opportunity pivoted **QLC → TLC/KV-cache** because NAND
  prices rose **5–10X** in a year and 2-terabit QLC dies are too constrained.
  TLC qualification started **a quarter early**; two customers in production,
  **five more Tier-1 CSPs** (3 Asia, 2 US) later this year.

| Metric | Q2 | Guide / note |
|---|---|---|
| Revenue | **$451.0M** (+127.0% YoY, +31.8% QoQ) | 5th straight YoY acceleration; sequential dollars $64M → **$109M** |
| Q3 guide | $519–541M → **+15–20% QoQ, +114–124% YoY** | a **23% beat** to the $431M estimate, though both rates moderate |
| Ferri & Boot Drive | **+110–115% QoQ, +1,690–1,695% YoY** | after +205–210% QoQ in Q1; Ferri automotive ~**30% of revenue** (was 4% YoY) |
| SSD controllers | +5–10% QoQ, +50–55% YoY | rebound from −5–10% QoQ |
| eMMC+UFS | +15–20% QoQ, +95–100% YoY | **decelerating** from +30–35% / +140–145% |
| GAAP EPS | $3.99 (+721.4%) | vs $2.04 est — includes a **$74.7M investment gain** |
| Adj EPS | **$2.43** (+254.3%) | ~20% beat; Q3's $2.28 estimate looks **too low** |
| GAAP GM | **50.2%** (+2.5 YoY, +3.0 QoQ) | Q3 guide 50.4% |
| GAAP OM | **22.4%** (+11.2 YoY, +7.1 QoQ) | Q3 guide **~25.1%** |
| Adj OM | 23.1% | Q3 guide **28%** |
| Net margin | GAAP 30.2% / **adj 18.4%** | GAAP inflated by the investment gain |
| OCF | **$(63.8M)**, (14.1%) margin | **3rd straight negative quarter**, worse than Q1's (9.1%) |
| FCF | $(71.5M), (15.9%) | |
| AR / Inv | $323.6M (+47% QoQ) / **$673.0M (+31% QoQ)** | |
| Cash | $181.8M incl. restricted — **$74.4M excluding** (was $135.7M) | **drew a $59.2M bank loan** — historically debt-free |

**Valuation:** 4X fwd sales (3yr avg 3.5X); **20X fwd earnings** (TTM 26X, 3yr avg
31X) — a premium to memory peers.

**Risks:** **the balance sheet is where to nitpick** — negative OCF three quarters
running, falling cash, first debt. NAND supply constraints could cap boot drive
shipments or delay MonTitan.

---

#### STX — Seagate (pricing power up, cost per TB down)

- Price per exabyte **+~10%** in the June quarter; guidance implies **+20% YoY**
  by September.
- **218 exabytes** shipped in Q4 (+34% YoY), **195 EB** from data centers — with
  drive-**unit** output essentially **flat**. Growth comes from more heads, more
  disks per drive, more capacity per disk.
- 3TB → **4TB per disk** (~⅓ more capacity at the same disc count) boosts margin.
- **HAMR** (heat-assisted magnetic recording) — a laser heats a spot so a single
  bit's polarity can flip, enabling large areal-density gains.

| Platform | Status |
|---|---|
| Mozaic 3+ | Qualified and in production at **all** major cloud customers |
| Mozaic 4+ | Ramping with the **two largest** global CSPs |
| Mozaic 5+ | Qualification shipments **late CY2027** |

| Metric | FQ4 | Guide / note |
|---|---|---|
| Revenue | **$3.63B** (+48.5% YoY, +16.6% QoQ) | above the $3.55B guide; **fastest sequential since 2012** |
| FQ1 guide | **$4.1B ±$100M** → ~**+56% YoY**, +13% QoQ | vs $3.75B estimate |
| Data Center | **$2.93B** (+57.4% YoY, +17.3% QoQ) | |
| Nearline exabytes | +11.4% QoQ, +42.3% YoY | **15-point delta** vs revenue growth (was 9 pts) → **pricing power strengthening** |
| GAAP EPS | **$5.58 (record)** (+149%) | |
| Adj EPS | **$5.71 (record)** (+121%) | 12%+ beat; FQ1 guide **$7.30 ±0.20 → +179% YoY** vs $5.85 est |
| GAAP GM | **52.3% (company record)** (+14.9 YoY, +5.8 QoQ) | |
| GAAP OM | **43.0% (record)** (+19.8 YoY, +10.9 QoQ) | opex **−2% QoQ** while revenue +17% |
| Adj OM | 44.6% | FQ1 guide **50%** — an aggressive +5.4 pts |
| Net margin | GAAP 35.7% / adj 36.3% | |
| OCF | $1.3B (36.0%), +157% YoY | FY26 **$3.67B (30.1%)**, >3X |
| FCF | $1.1B (30.8%), +2.5X | FY26 **$3.1B (25.5%)**, ~4X (from 9%) |
| Cash / debt | $1.7B / $3.6B → **net debt $1.9B** | gross debt −$302M in Q4, **−$1.4B** for the year |
| Net leverage | **0.4X** (from 1.8X) | +$1B high-yield notes extinguished July 2026; converts retired Sept |

**Valuation:** **8X fwd sales** (TTM 12X, 3yr avg 5X) — a substantial premium to
history and to WDC 7X / SNDK 4X. **18X fwd earnings** (TTM 48X), vs WDC 17X.

**Risks:** HAMR qualification/ramp delays; margins depend on sustained pricing
and mix; still a cyclical storage business.

---

### 10.3 Semiconductors and equipment

---

#### AMD — Venice CPUs and the Helios ramp

- *"Customer demand for **Venice** is stronger than for any prior EPYC generation."*
  Zen 6, **2nm**. Claims **2X perf/watt** vs leading x86, **3.3X** vs leading Arm.
- *"When we go into Venice, the **workloads actually expand**."* ASP growth from
  higher core counts — *"expect both unit and ASP growth."*
- **Helion** (announced last week): CPU rack for agentic workloads — up to
  **32,768 CPU cores** CPU-only, or a config with 16 MI355X GPUs. First racks
  **in November**.
- **Helios** = Venice CPUs + MI450 GPUs + Pensando networking + ROCm. Up to **15%
  more throughput at the same rack power**, **30% more tokens per dollar**.
  Ramps end of Q3, "much more substantial in Q4."
- 2027: data center segment to **"more than double"**, *"well over 100%"* — with
  Helios carrying growth **above** the ~70% expected from CPUs.
- Deployments: first **1GW** of OpenAI's 6GW deal ships 2H; first **1GW** with
  **Anthropic** early 2027.

| Metric | Q2 | Guide / note |
|---|---|---|
| Revenue | **$11.54B** (+50.1% YoY, +12.5% QoQ) | **fastest YoY in 4 years**; 2% beat. Prior Q2 seasonality was +6.6% / +3.3% |
| Q3 guide | **$13B ±$300M** → ~+41% YoY, +13% QoQ | a **9.5-pt YoY deceleration** right before the Helios ramp — a possible market concern |
| Data Center | **$6.72B** (+107.3% YoY, +16.3% QoQ) | a **50-point** YoY acceleration from +57.3% |
| GAAP EPS | $1.38 (+156%) | vs $1.05 est — includes a **$483M** investment gain |
| Adj EPS | **$1.66** (+246%) | vs $1.61 est; **200+ point** acceleration from +43% |
| Q3 est | GAAP $1.34 (+79%) / Adj $1.86 (+55%) | **a bump in the road** before Helios |
| Q4 / Q1'27 est | Adj **$2.63 (+72%)** / **$2.83 (+107%)** | reacceleration |
| GAAP GM | 53.8% | YoY comp skewed by last year's export charge |
| Adj GM | 56.2% | Q3 guide **~56% (flat)** |
| GAAP OM | 17.3% (+2.9 QoQ) | |
| Adj OM | 26.8% (+2.0 QoQ) | Q3 guide **27.9%** — slight leverage |
| Net margin | GAAP 19.9% / adj 23.9% | |
| OCF | $2.37B (**20.5%**, from 28.8%) | a **$1.25B** AR swing |
| FCF | $1.56B (**13.5%**, from 25.0%) | capex more than doubled QoQ to **$808M** |
| Cash+STI / debt | $13.11B / $3.23B | |
| AR / Inv | **$7.28B (+20.7% QoQ)** / $8.47B (+5.3%) | inventory **not** building ahead of Helios… |
| **Purchase commitments** | **$30.28B** (from $25.66B in Q1, itself 2X QoQ) | **$17.39B due in the rest of 2026** — preparing for a sharp 2027 volume ramp |

**Valuation:** **17X fwd sales** (TTM 22X, 3yr avg 10X) — premium to **NVDA 13X**,
in line with MRVL 17X. **71X fwd earnings** (TTM 139X, 3yr avg 221X) — versus
**NVDA 23X** and MRVL 56X.

**Risks:** Helios execution (yields, software maturity, deployments); **$30B of
purchase commitments** = inventory/financial exposure if demand disappoints;
71X vs Nvidia's 23X leaves little room for delay.

---

#### LRCX — Lam Research

- TSMC raised 2026 capex **+$8B to $62B** in Q2 → Lam is the downstream WFE
  beneficiary.
- **NAND is the under-modelled piece.** Lam said in early 2025 that ~**$40B** of
  conversion spending was needed to take installed NAND capacity past 200 layers.
  Now: *"the majority of spending occurring **before the end of calendar year
  2027**."*
  - Conversions/upgrades are quickest; **greenfield** is needed for real growth
    and is *"materializing as a significant opportunity now on the revenue side."*
  - **The disconnect:** consensus models only **$6.78B** of cumulative Lam NAND
    revenue through end-2027 — ~**17% share** of the $40B, **below** Lam's ~**25%**
    deposition/etch share.
  - But NAND is only the **third** fastest grower in 2027, behind **DRAM** and
    **Foundry**.
- **HBM4** (+50% layer count) raises content: *"the higher stack requires more
  equipment."* **Akara** etch tool gaining momentum in advanced DRAM — installed
  base to **double again YoY in 2027**. **VECTOR** hard mask deposition +
  co-optimized conductor etch → better yield and **up to 20% cost savings**.

| Metric | FQ4 (June) | Guide / note |
|---|---|---|
| Revenue | **$6.72B** (+30.0% YoY, +15.1% QoQ) | 3rd straight YoY acceleration; guide was $6.60B |
| FQ1 guide | **$8.1B** (mid) → **+52% YoY, +20.5% QoQ** | **$1.4B** sequential; fastest growth since early 2021 |
| FY27 est | **$34.79B (+49.8%)** | |
| Systems | $4.25B (+23.6% YoY, +13.9% QoQ) | |
| — Foundry | $1.87B, **44%** of Systems | **+4.6% YoY**, sharply decelerated (was +38.3%, +115.6%) |
| — DRAM | $977M (**+103% YoY**, −3% QoQ) | after two quarters of >30% QoQ |
| — NAND | $977M (**+118.3% QoQ**, +5.3% YoY) | ~23% of Systems |
| GAAP EPS | $1.81 (+34.6%) | |
| Adj EPS | $1.82 (+36.5%) | **slowest EPS growth of FY26**; FQ1 guide **$2.15 → ~+70%** |
| FY27 est | GAAP $8.77 (+52.4%) / Adj $8.62 (+48.1%) | ≈ revenue growth → **minimal margin expansion** |
| GAAP GM | **51.7%** (+1.7 YoY, +1.9 QoQ) | **highest since 2007** |
| GAAP OM | **37.4% (record)** (+3.7 YoY) | |
| Adj OM | **38.4%** | beat the 36.5% guide; FQ1 guide **39.5% ±1%** |
| GAAP net margin | 33.9% | |
| OCF | $1.46B (21.7%) | FY26 $5.86B (25.2%), down from 33.5% — **FY25 had an outsized working-capital benefit** |
| FCF | $1.27B (18.9%) | FY26 $4.89B (21.1%) |
| Cash / debt | $5.60B / $3.73B (from $4.48B) | |
| Inventory | $4.3B (+$276M QoQ) | |

**Valuation:** 10X fwd sales (TTM 15X, 3yr avg 9X) — in line with AMAT 10X, below
KLA 12X, above ONTO 8X. **29X fwd earnings** (TTM 48X, 3yr avg 31X) — favourable
vs AMAT 33X and KLA 31X.

**Risks:** WFE cyclicality; a slower-than-expected $40B NAND conversion; China
export restrictions; competition in deposition/etch.

---

#### TSM — TSMC *(honorable mention)*

- **3nm capacity**, not CoWoS, is arguably the sharper constraint.
- HPC revenue **+22% QoQ (NTD)**, after +20% in both Q1 and Q2. In USD: **~$21.9B
  → ~$31.8B** across FY26.
- No formal update to the **"high-50%" AI accelerator CAGR**, but CEO C.C. Wei
  hinted it *"probably will be higher."* Excludes **AI CPUs**.
- Tone shift: *"no intention to leave any food on the table for others"* on
  leading-edge capacity and packaging — versus last quarter's welcome of Intel
  EMIB-T flexibility.
- **20 fabs in progress.** Three **new 3nm fabs** despite a policy of not adding
  capacity past target: Taiwan **1H 2027**, Arizona **Q3 2027**, Japan **1H 2028**.
  **Texas** under consideration.
- Capex: FY26 **$60–64B** maintained → analysts **$80–90B (2027)**, up to **$105B
  (2028)**. *"Tool supply is tight"* and *"tool price inflation is real."*
- **2027 price increases** expected: **5–10%** advanced node, **10–15%** advanced
  packaging.

| Metric | Q3 | Guide / note |
|---|---|---|
| Revenue | **$46.7B** (+41.2% YoY, +16.2% QoQ) | above the $44.6–45.8B guide |
| Q4 guide | **$52.3–54.3B** → **+58% YoY, +14.1% QoQ** | vs $50.5B est |
| FY26 | **+44%** (raised from "slightly above 40%") | ≈ **$176.3B** |
| HPC | ~$31.8B, **68%** of revenue (+68% YoY, +20% QoQ USD) | mix +11 pts YoY |
| Smartphone / Auto / IoT | 21% / 4% / 4% | +10% / +23% / +5% QoQ NTD |
| Advanced nodes | **79%** of revenue (from 77%) | **2nm 8%** (from 3%, capacity doubles 2027); 3nm 33% (+3); 5nm 28% (−5) |
| GAAP EPS | $4.60 (+57.5%) | Q4 est $5.15 (+64%) |
| GAAP GM | 66.4% (+7 YoY, **−2 QoQ**) | 2nm ramp; guide 65–67%. Overseas fabs = **2% headwind**, widening to **3–4%** |
| GAAP OM | 58.7% (+8 YoY, −1.6 QoQ) | guide 56.5–58.5% |
| GAAP net margin | 51.5% | |
| OCF / FCF | $28.8B (61.6%) / $14.9B (31.9%) | capex $15.7B → **$13.8B** QoQ |

**Valuation:** **12.4X fwd sales** (5yr avg 10.3X; hit 14.5X in early October).
**23.6X fwd earnings** (5yr avg 22.5X) — shares consistently find resistance at
**30X forward**.

**Risks:** China/Taiwan geopolitical concentration; 3nm capacity as a brake on the
whole AI buildout; advanced packaging limits.

---

#### AVGO — Broadcom *(honorable mention)*

**#2 in AI chips and #2 in AI networking**, both behind Nvidia.

New guidance — **~4X AI semiconductor revenue in two years (~100% CAGR)**:

| Year | AI semi revenue |
|---|---|
| FY26 | **$58B** (raised $2B, +186%) |
| FY27 | **$115B** |
| FY28 | **$230B** ("line of sight") |

Customer deployments named by CEO Hock Tan:

- **Anthropic** — 5GW of TPU 8i in 2027; **incremental 10GW in 2028**; on track to
  become **Broadcom's largest XPU customer in 2027**
- **OpenAI Jalapeno** — 1.3GW in 2027; 5GW in 2028 (Jalapeno + 2nd-gen)
- **Meta MTIA** — 3GW through 2028

Analysts count **up to 30GW cumulative** across 2027–28 = **$350B of shipment
visibility**. Concentration risk: OpenAI + Anthropic ≈ **21GW** of it.

**Networking:** Tomahawk6 is the fastest-ramping switch ever; **Tomahawk7** is the
industry's first **204.8T** Ethernet switch; **Tomahawk Ultra** the first scale-up
Ethernet switch (ramping Q3 and through 2027). EML, CW and VCSEL / InP factories
(US + Singapore) **more than tripling YoY**.

**Circular financing (the stated risk):** AI XPV platform with Blackstone and
Apollo designed for **20GW** of external compute. First **$35B** tranche in June
for 1GW of Anthropic. Now reportedly seeking **$60B** more for Anthropic and
**$50B** for OpenAI → **$145B lined up**.

| Metric | Q3 | Guide / note |
|---|---|---|
| Revenue | **$29.59B** (+85.5% YoY, +33.4% QoQ) | from +47.9% / +14.9% in Q2 |
| Q4 guide | **$34.8B** → +93% YoY, ~+18% QoQ | |
| Semiconductors | $20.89B (**+127% YoY**) | Infrastructure Software $8.75B (+29%) |
| AI semis | **$16.7B** (+221% YoY, +54% QoQ) | from +143% / +29% |
| — XPUs | ~$12.2B, **~73%** (shipments **+3.5X YoY**) | |
| — AI networking | ~$4.5B (**+2.5X YoY**) | expected to grow *"just as fast as XPUs"* |
| Q4 AI semis | **$21.7B** (+236% YoY, +30% QoQ) | both XPU and networking **triple YoY** |
| GAAP EPS | $2.68 (+215%) | FY26 est $9.25 (+94%) |
| Adj EPS | $3.32 (+96%) | vs $3.24 est; FY26 est $11.66 (+71%); **FY28 guided >$30** |
| GAAP GM | 69.1% (+2 YoY) | Adj GM **75% (−3 YoY)** on rising XPU mix |
| GAAP OM | **53.9%** (+17 YoY) | Adj OM **67.9%**; adj operating income a record **$20.1B** |
| — Semis adj OM | 61% (+4 YoY) | — Software **84%** (+6 YoY) |
| Net margin | GAAP 44.2% / adj 55.3% | |
| OCF / FCF | $14.2B (48%) / $13.7B (46%) | |
| Cash / debt | ~$24B / **$59.4B** | |

**Valuation:** **14.9X fwd sales** (5yr avg 15.6X) and **28.4X fwd earnings**
(5yr avg 32.1X) — *below* its own averages on both.

**Risk (named):** **circular financing.** $145B lined up or being arranged, and
the figure may keep rising.

---

### 10.4 Energy and power

---

#### BE — Bloom Energy ("chips without power are inventory, not intelligence")

The author's **#1 stock of 2026**.

- On rivals' multi-year backlogs: *"Grid operators quote years, timelines on which
  billion-dollar compute clusters go obsolete in a warehouse… **We think a
  four-year backlog is not a trophy.**"*
- The opportunity cost, quantified: a 1GW data center generates **$12–24B of
  revenue per year**. Pulling power in **within a month** is **$1–2B** of revenue
  the customer would not have had, at **40–50% gross / 20–25% net margin**.
- *"All the major US hyperscalers and over a dozen US neoclouds, AI labs, and
  colocation data center operators have validated and approved our power
  solutions."* (Last quarter only Oracle, Brookfield and AEP were named, with
  "more than half" of DC backlog from other hyperscalers/neoclouds/colos.)
- **Brookfield:** project capital commitment **$5B → $25B** in nine months (5X).
  Brookfield **owns the energy servers**; customers pay over time via PPA
  ($/kWh), capacity agreement (installed capacity) or equipment lease — **no
  large capex sale**, removing a financing bottleneck.

| Metric | Q2 | Guide / note |
|---|---|---|
| Revenue | **$1.065B** (+165.5% YoY, +41.8% QoQ) | first $1B quarter; **28.8% beat** vs $827M. Q1 was −3.4% QoQ |
| FY26 guide | **$3.9–4.2B (~+100%)** | raised from +80%, originally **+60%** |
| Product | **$935.4M** (+215.4% YoY, +43.2% QoQ) | **88%** of revenue. Q2'25 was $296.6M — this quarter's **sequential gain alone** ≈ last year's whole quarter |
| GAAP EPS | **$0.62** (est $0.25 → **148% beat**) | from $(0.18); +$14.1M non-operating income |
| Adj EPS | **$0.78** (+680%, est $0.41 → **90.2% beat**) | FY26 guide raised to **$2.55–2.85** (from $1.85–2.25) → needs **$1.48 in 2H** |
| Adj EBITDA | $253.4M (**23.8%**, ~6.1X YoY) | from 10.3% YoY, 19% in Q1 |
| GAAP GM | 33.4% (+6.7 YoY, +3.4 QoQ) | **Product GM 36.5% GAAP / 37.2% adj** — the main lever |
| Adj GM | 34.3% | |
| GAAP OM | **17.1%** (from −0.9%) | |
| Adj OM | **22.5%** (+15.4 YoY, +5.2 QoQ) | **adj opex +21.7% vs revenue +165.5%** |
| FY26 adj op income | **$800–900M** (4X YoY from $221M) | ~21% margin, +10 pts |
| Net margin | GAAP 18.4% / adj 23.3% | |
| OCF | **$226.4M (21.3%)** | a **$439M swing** from $(213.1M); 4th straight positive quarter |
| FCF | $174.8M (16.4%) | from $(220.4M), (55%) |
| Cash / debt | $2.69B / $2.48B | **modest net cash** for the first time in recent memory |

**Valuation:** **16X fwd sales** (TTM 21X, 3yr avg **7X**) — a substantial premium
to GEV 5X, CEG 3X, VST 2X. **83X fwd earnings** (TTM 307X) vs GEV 60X and
traditional power producers at 17–22X.

**Risk (stated in full):** *"Bloom's premium valuation leaves little room."*

---

#### GEV — GE Vernova

The constraint **is** the thesis: gas capacity is **20 GW/yr now, 30 GW by 2030**,
so growth comes from **price, services, time-to-power products and
electrification mix** — not from shipping more turbines.

- 1H 2026 equipment orders priced **20% higher** than Q4 2025. 2H expected at the
  **higher end of the 10–20 point** range vs Q4 2025 orders.
- **56 GW** under SRAs as of Q1; pricing power becomes visible as slot reservation
  agreements convert at higher $/kW, improving **backlog margins**.
- **Time-to-power products:** aeroderivatives and combined-cycle (steam turbines
  capturing exhaust heat) — both lifted $/kW this quarter.

| Metric | Q2 | Guide / note |
|---|---|---|
| Revenue | **$11.10B** (+22% YoY, **+12% organic**, +18.9% QoQ) | fastest in 5 quarters; **Wind −10% YoY** is the drag |
| FY26 guide | **$45.5–46.5B** (+20.7% mid) | raised 2nd straight quarter |
| — Power organic | **18–20%** (from 16–18%) | — Electrification **$14.5–15B** (from $14–14.5B); Wind down low-double-digits |
| Orders | **$24.2B, +88% organic** | from +71% in Q1 |
| **Backlog** | **$176.3B (record)**, +$13.0B QoQ, **+37% YoY** | on track for **$200B by 2027**. 3rd straight quarter backlog growth beat revenue growth by **15+ pts** |
| Gas equipment + SRAs | **100 GW → 116 GW** | **≥125 GW by YE26** (was 110 GW); 52 heavy-duty (15 HA) + 61 aeroderivative orders |
| Power revenue | $5.48B (+14% YoY/organic) | Gas Power $4.43B (+13%); **Q3 guide +17–19% organic** |
| Power orders | **$16.7B, +134% organic** (equipment **+259%**) | 20 GW signed in Q2 = **a full year of capacity**; backlog +9 GW to 53 GW, SRAs +7 GW to 63 GW |
| Power backlog | $111.6B (equipment **+143% YoY** to $39.3B) | |
| Electrification revenue | **$3.64B** (+68% YoY incl. Prolec, **+29% organic**, +22.9% QoQ) | Q3 guide $3.8–4.0B (**+50% YoY** — a steep decel; +6.6% QoQ) |
| Electrification orders | **$6.3B, +66% organic** | book-to-bill **~1.7X**; equipment +72% to $5.67B; backlog +69% YoY to $40.6B |
| Data center orders (Elec.) | **>$5B YTD** | **more than double** all of 2025 |
| GAAP EPS | $2.47 (+33%) | **a 22.3% MISS** vs $3.18 — flagged **$100–200M of tariff costs** for the rest of the year |
| Q3 / Q4 est | $4.31 (+163%) / $6.01 (−55% vs a tax-benefit comp) | |
| Gross margin | 21.3% (+100 bp YoY, +220 bp QoQ) | |
| Operating margin | **5.9%** (+170 bp YoY, +400 bp QoQ) | still thin |
| Adj EBITDA | $1.25B (+62%), **11.3% margin** (+280 bp YoY, +170 bp QoQ) | **highest since the spin-off** |
| OCF / FCF | **$5.49B (49.5%)** / **$5.11B (46.0%)** | vs $367M / $194M YoY; ~**787%** FCF conversion |
| FY26 FCF | raised to **$11.5–12.5B** (from $6.5–7.5B) | **but 1H was $9.9B** → only **~$2.1B** left for 2H (~$1B/quarter) as SRAs convert to orders |

**Valuation:** 5X fwd sales (TTM 6X) — in line with NEE 5X, premium to CEG 3X,
VST 2X. **60X fwd earnings** — below BE 83X, well above power producers at 17–22X.

**Risks:** the **capacity ceiling** (20 GW now, 30 GW not until 2030) delays
backlog conversion; tariffs and supply chain pressure margins.

---

### 10.5 Software

---

#### NET — Cloudflare ("humans will be a rounding error on the internet")

- **Q2 was the first time >50% of traffic on Cloudflare's network was not human.**
- Cloudflare handles **~20% of the internet**, **mid-30%** of the top 10,000 sites
  → agentic traffic routes through its reverse proxy and CDN. More traffic → more
  CDN/proxy/Workers demand, and more (potentially malicious) traffic → more
  security revenue.
- **The business model argument:** *"The traditional first-generation clouds are
  in the business of buying a server and then trying to… get 5 turns of revenue
  off of it. We're in a very different business, where we're selling actually
  work getting done."* And: *"if you're basically letting an AI company use your
  balance sheet, and your credit rating in order to buy servers that are the same
  as everybody else's servers… that's just not an attractive business for us."*
- **Isolates, not containers:** *"even things like containers are too heavy"* —
  lightweight sandboxing across a distributed network, matching how agents
  retrieve data. **Lower capital intensity** than hyperscalers.

| Metric | Q2 | Guide / note |
|---|---|---|
| Revenue | **$696.1M** (+35.9% YoY, +8.8% QoQ) | 4.5% beat; **fastest since Q1 2023** |
| Q3 guide | $736–737M → +31% YoY, +5.8% QoQ | still **~$43M** sequential vs $56M in Q2 |
| FY26 guide | $2.864–2.87B (**+32.2%**) | below 1H's 33.5%/35.9% → likely **conservative** |
| NRR | **120%** (+2 QoQ, +6 YoY) | hasn't meaningfully broken above 120% |
| GAAP EPS | $(0.48) | **$150.7M restructuring/severance**, not deterioration |
| Adj EPS | $0.29 (+38.1%) | decel from +56.3%; Q3 guide $0.34 (**+25.9%**); FY26 $1.25–1.26 (~+35%) |
| GAAP GM | 71.8% (**−3.1 YoY**, +<1 QoQ) | a **multi-quarter** YoY contraction; adj GM 73.1% |
| Adj OM | Q3 guide **17.6%** (+3.8 QoQ) | FY26 guide **15.5%** vs 11.4% / 13.8% in 1H |
| OCF | $117.6M (**16.9%**, −7.8 QoQ, −2.6 YoY) | lowest since Q1 2023 |
| FCF | $56.4M (**8.1%**, −5 QoQ) | **capex guided 14–15%** of revenue but was only 7% / 9% in 1H → **2H step-up** |
| Cash / converts | $1.66B / $3.27B | |

**Valuation:** **37X fwd sales** (TTM 42X, 3yr avg 26X) — premium to PANW 20X,
ZS 9X; in line with CRWD 39X. **237X fwd earnings** vs CRWD 182X, PANW 83X.

**Risks:** AI traffic is accelerating but **AI revenue is not quantified**; gross
margins under pressure; FCF margin in single digits with capex rising in 2H.

---

#### CRWD — CrowdStrike (AIDR)

- **AIDR** (AI Detection and Response) — a **new runtime security category**:
  where AI agents run, what exposures they create, *"where those agents are
  calling out to"* — sandbox, endpoint or cloud.
- **AIDR ARR tripled QoQ** in Q2 after **+250% QoQ** in Q1 → **~10X in two
  quarters**. *"Almost the results sort of parallel some of the frontier labs
  growth."*
- Monetized **separately and incrementally** to core EDR. **Token-based pricing**
  gives customers visibility into **token usage and cost** — a combination few (if
  any) security vendors offer.
- **Falcon Flex** = single agent, single console → adding a module is a **toggle**,
  not a deployment.
- Caveat the author states plainly: **headline numbers do not yet show the AI
  acceleration**; QoQ growth is mid-single-digit.

| Metric | FQ2 | Guide / note |
|---|---|---|
| Revenue | **$1.47B** (+25.8% YoY, +6.2% QoQ) | roughly matching FQ1 |
| FQ3 guide | ~$1.53B (+23.6% YoY, +3.7% QoQ) | mid-20s may be the near-term plateau |
| FY27 guide | **$5.99–6.01B (+24.7%)** | raised from $5.91–5.96B |
| **Net new ARR** | **$333M (all-time record)** | **$48M beat** vs $285M guide; **+51% YoY** (+19 pts) |
| FQ3 NNARR guide | **$343–347M (+30%)** | another record |
| FY27 NNARR | **$1.355B (+34%)** | raised **+$116M / +1,150 bps** from the initial +22.5% |
| Ending ARR | $5.84B (+25%) | FY27 guide **$6.60–6.61B (+26%)** |
| GAAP EPS | $0.01 (from $(0.07)) | |
| Adj EPS | $0.31 (+35%) | beat $0.29; FQ3 guide $0.31 (+29%) → **minimal OM expansion ahead** |
| GAAP GM | 75% (+1 YoY) | |
| GAAP OM | (2%) (+7 YoY) | adj OM **25%** (+3 YoY); FQ3 guide ~24.5% flat |
| GAAP net margin | 0.4% | |
| OCF | $530.3M (**36%**, from 28%) | |
| FCF | **$377M (record), 26%** | from 24% |
| Cash / debt | **$5.01B / $746M** | |

**Valuation:** **39X fwd sales** (TTM 44X, 3yr avg 24X) — in line with NET 37X,
premium to PANW 20X / ZS 9X. **182X fwd earnings** vs NET 237X, PANW 83X, ZS 43X.

**Risks:** AIDR's contribution to consolidated revenue is **unproven**; the July
2024 outage still carries reputational and legal risk; elevated SBC pressures
GAAP; little room at 39X/182X.

---

#### PLTR — Palantir *(honorable mention)*

The comparison that matters now is **Palantir vs the AI labs** — they compete for
the same enterprise budgets.

- The attack on consumption pricing: *"Enterprises that are not using Palantir are
  seeing their **token meters spinning endlessly just to get slop** without any
  correlation to value. This token model may be working for the labs, but it is
  not working for anyone else."*
- Palantir prices on **"economic value"** — enterprise agreements across
  workflows, not raw token usage. *"The market has created far more intelligence
  than it has converted into value. A more powerful model does not solve this."*
- **Model-agnostic by design:** Nvidia's open-source **Nemotron Ultra** reportedly
  outperformed a leading frontier model. *"It's not about being beholden to one
  model."* The author notes this view is **self-serving** — commoditized LLMs push
  advantage to the orchestration layer, which is exactly where Palantir sits.
- No training on customer datasets → a data-safety claim vs the R&D labs.

| Metric | Q2 | Guide / note |
|---|---|---|
| Revenue | **$1.94B** (**+92.8% YoY — a record**, +18.6% QoQ) | 7.9% above guide, 6.8% above consensus; an **80-point acceleration** from the 12.8% trough three years ago |
| Q3 guide | **$2.162B** (+83.1% YoY, +11.7% QoQ) | last quarter's guide also implied a decel that didn't happen |
| FY26 guide | **$8.154B (+82.2%)** | raised from $7.656B (+71.1%) — but implies a **20-point decel into Q4** toward ~72%, so possibly **conservative** |
| Commercial | **$945M** (+109.5% YoY, +22.1% QoQ) | **48.8% of revenue** — on the cusp of passing Government |
| US Commercial | **~+150% YoY** | Commercial now outpaces Government by **30 pts** (was 20) |
| GAAP EPS | $0.41 (+215%) | SpaceX stake: **+$0.03 GAAP / +$0.02 adj** |
| Adj EPS | $0.41 (+156%) | beat $0.35; Q3 implies GAAP $0.38 (+111%), adj $0.41 (+91%) |
| **Rule of 40** | **155%** (+61 pts YoY) | |
| GAAP GM | 84.7% (−2 QoQ, +4 YoY) | adj 86.3%; softness from **taking on cloud hosting for a government customer** |
| GAAP OM | **47.1%** (+20.3 YoY) | approaching 50% |
| Adj OM | **61.7%** (+15.4 YoY) | 2nd straight quarter >60%; Q3 guide 59.9% |
| Net margin | GAAP 54.9% / adj 54.1% | |
| OCF | **$1.22B (62.8%)** (+7.7 QoQ, +9.1 YoY) | |
| Adj FCF | $1.22B (**63.1%**) | FY26 raised to **$4.5–4.7B** (from $4.2–4.4B), a 56.4% margin |
| Cash+securities / debt | **$9.4B / zero** | |

**Valuation:** **57X fwd sales** (TTM 81X, 3yr avg 60X) — vs NET 37X, DDOG 21X,
SNOW 18X. **120X fwd earnings** (TTM 166X, 3yr avg **317X**) — above DDOG 104X,
NOW 36X.

**Risks:** hyperscalers and enterprise software building their own AI platforms;
sustaining triple-digit US commercial growth gets harder with scale; 57X/120X
leaves little room.

---

## 11. Cross-cutting observations worth questioning later

1. **The cheapest things in the report are the memory names** — MU, SK Hynix and
   SNDK all at ~4X sales and **5–6X earnings** — precisely because the market is
   pricing a cycle top. The author argues agentic concurrency breaks the cycle.
2. **The most expensive are ALAB (27X/73X), PLTR (57X/120X), NET (37X/237X) and
   CRWD (39X/182X).** Each risk paragraph says some version of "little room."
3. **COHR is the odd one out in optics:** 4X sales and 26X earnings vs LITE's
   12X/40X, explained by materially lower margins (adj OM 21.8% vs 36.6%) and
   **$1B negative FCF**. The thesis is margin convergence.
4. **Cash flow is the tell in several names.** Negative or deteriorating FCF at
   AAOI, COHR, SIMO (3 straight negative OCF quarters, first-ever debt draw),
   NET (single-digit FCF margin with capex about to step up). GEV's headline FCF
   raise leaves **only ~$1B/quarter for 2H**.
5. **Gross margin direction splits the networking group.** LITE expanding
   (+12.6 YoY adj), ALAB **contracting two quarters running**, AAOI flat-to-down,
   MXL flat.
6. **GAAP-vs-adjusted gaps worth remembering:** LITE's $(84.65) GAAP EPS is a
   $7.76B non-cash convert charge; MXL's 24-point gap is ~$36.5M of SBC;
   MTSI, SIMO, ALAB and AMD all had one-off investment/tax gains inflating GAAP.
7. **Concentration risk is named twice:** Broadcom (OpenAI + Anthropic ≈ 21GW of
   30GW, plus **$145B** of arranged circular financing) and AMD (**$30.28B** of
   purchase commitments, $17.39B due in the rest of 2026).
8. **The two "constraint migrates" claims:** CoWoS → SoIC → reticle → substrates;
   and EML → CW lasers (where AAOI's vertical integration and Lumentum's premium
   both get tested).

---

## 12. Disclaimer carried from the source

> I/O Fund conducts research and draws conclusions for the Fund's positions, then
> shares that information with readers. This is not a guarantee of a stock's
> performance. I/O Fund analysts may hold positions in securities mentioned.
> Past performance is not indicative of future results. Not financial advice.

**Prior reports referenced:** Top 20 Q3 2026, Top 15 Q2 2026, Top 15 Q1 2026,
Top 15 Q4 2025, Top 15 Q3 2025.
