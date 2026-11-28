# Operating-room planning & scheduling: literature notes

Side research started 2026-11-28 around Hashemi Doulabi, Rousseau & Pesant (2016), IJOC 28(3):432–448 (the "2016 paper"),
which this repository reproduces. Companion index: [`papers.json`](papers.json) (machine-readable).

## How much to trust each entry

| Tag | Meaning |
|---|---|
| **[P]** | Primary text read in this project. Only the 2016 paper. |
| **[S]** | Seen only through a web-search summary of the abstract or catalogue record. Authors, venue and claimed numbers are the sources' words. The full text was **not** read. |
| **[R]** | Appears only in a reference list or snippet of another source. Treat as a lead. |

Why so little is [P]: the web-fetch tool cannot reach arxiv.org (it bypasses the session proxy), and a shell `curl` to download the PDFs was refused by the permission classifier. Everything below marked [S] or [R] should be confirmed against the paper before it is cited. See "Papers to obtain" at the end.

## 1. Where the 2016 paper sits

**What it does [P].** Integrated OR planning and scheduling (assign cases to day and room *and* time them), open scheduling, deterministic durations, infection cleaning, surgeon daily limits and no-overlap. Dantzig–Wolfe master over room-day columns, CP pricing subproblem, dominance rules, knapsack-based infeasibility detection, lifted cover cuts, three branching rules. Average gap 2.81% on generated instances of 40–120 cases.

**Future work it names [P]:** (1) uncertain durations and emergency arrivals (it notes only Batun et al. 2011 covered open scheduling, with at most 11 surgeries); (2) integrating the tactical level (block counts and times) with the operational plan.

**Lineage.**
- Conference precursor: CPAIOR 2014, LNCS 8451, pp. 455–463, "A constraint programming-based column generation approach for operating room planning and scheduling" [S].
- Earlier building blocks cited by the paper [P]: Roland et al. 2006/2010; Marques, Captivo & Pato (OR Spectrum 2012, integer program); Fei, Meskens & Chu 2006/2009/2010; Vijayakumar et al. 2012.
- Surveys: Cardoen, Demeulemeester & Beliën (EJOR 2010), Guerriero & Guido (2011) [R].

## 2. Developments since 2016 (grouped by theme)

### 2.1 Direct competitors on the same problem

| Work | What it claims | Tag |
|---|---|---|
| **Roshanaei & Naderi, EJOR 293(1):65–78 (2021)**, "Solving integrated OR planning and scheduling: logic-based Benders decomposition versus branch-price-and-cut" (DOI 10.1016/j.ejor.2020.12.004) | Names the 2016 hybrid CP/IP + branch-price-and-cut as the state of the art. New MIP model: average gap 3.84% vs 11.84% for the CP-IP model. Best Benders (LBBD / branch-and-check) variant: **0.88% vs 2.81%** for the 2016 algorithm, "using the same dataset". Both baseline numbers match the 2016 paper's Tables 6–7 (2.81% and 11.84%). | [S] |
| **Dolatkhah, Hashemi Doulabi, Rei & Gendreau**, arXiv 2604.22977 (2026), "A reinforcement-learning-based column generation algorithm for integrated OR planning and scheduling" | New MIP covering mandatory and elective surgeries, overtime, and surgeons' daily availability; column generation whose **initial columns come from a hybrid reinforcement-learning + genetic algorithm**. A press release (Concordia) says the model chooses which rooms to open each day and start times, and may defer a few elective cases. One search snippet attributed "4.13% deviation, same-day emergency insertion" to this paper; that is **unverified and possibly mis-attributed**. | [S] |
| **Bargetto, Garaix & Xie, Computers & OR 152:106136 (2023)**, "A branch-and-price-and-cut algorithm for OR scheduling under human resource constraints" (HAL hal-04609352) | Adds nurses/human resources and sequence-dependent cleaning for infection levels. Pricing = single-room scheduling with time-dependent costs, solved by a **label-correcting algorithm** with label dominance and label bounds; **Benders-inspired cuts** tighten the LP. | [S] |
| **Müller, Moreno & Weiland**, Optimization Online (July 2026), "Exact branch-and-price algorithm for live OR reoptimization" | Live rescheduling after disruptions. **Bucket-graph labeling** pricing, two exact pruning methods ("surgical case blocking", problem-specific completion bounds) that preserve LP optimality. | [S] |
| **arXiv 2509.24806 (2025)**, "A bilevel approach to integrated surgeon scheduling and surgery planning solved via branch-and-price" | Hierarchical (bilevel) model with surgeons as stakeholders; Dantzig–Wolfe, branch-and-price. A related CTU Prague thesis frames group and individual surgeons with conflicting objectives. Abstract text was not retrieved. | [S]/[R] |
| **Akbarzadeh & Maenhout, EJOR 322(2):448–466 (2025)** | Patient planning and surgeon scheduling, branch-price-and-cut with an "intelligent branching scheme". | [R] |

### 2.2 Making branch-and-price faster with machine learning

- **Koutecká, Šůcha, Hůla & Maenhout, EJOR 320(2):328–342 (2025)**, "A machine learning approach to rank pricing problems in branch-and-price" [S]. A learned ranker uses the master solution to order the pricing problems; the abstract says it consistently beat traditional strategies in time, number of pricing problems solved, tree nodes and CG iterations on OR scheduling. Precursor: Václavík et al., EJOR 271(3):1055–1069 (2018), reuse of earlier pricing runs to shrink later ones [R/S].
- General ML-for-column-generation [S], from one search that returned no survey: RL multiple-column selection (Yuan, Fang & Song, AAAI 2024); ML column selection (Morabit, Desaulniers & Lodi; reported up to 30% faster on VRPTW); ML pricing heuristic for graph coloring (Shen et al.); learned dual prediction and stabilisation (Babaki et al.; arXiv 2405.11198; "Learning to control stabilization in column generation", arXiv 2604.23889; a "predictive dual smoothing" arXiv entry whose ID looks unusual and needs checking).

### 2.3 Uncertainty (the 2016 paper's Assumption 1)

- **Xu, Yin & Wang**, "Distributionally robust multi-period OR scheduling with multiple surgical disciplines under uncertain surgery durations" [S]. Distributionally robust chance constraints on session capacity, data-driven mean/covariance ambiguity set, converted to a MILP and solved by **branch-and-price-and-cut with bounded bidirectional dynamic programming pricing**; reported to beat CPLEX, a stochastic-programming counterpart and two heuristic rules. Venue and year were not shown.
- **Wang, Zhang & Tang, EJOR 314(2):509–522 (2024)** [S]: elective + emergency patients, uncertain durations, Wasserstein-type DRO, branch-and-cut, real data, reserved emergency capacity.
- Robust vs stochastic: Denton-style minimax models; Guo, Bodur, Aleman & Urbach, arXiv 1907.13265 (LBBD plus binary-decision-diagram Benders cuts for stochastic distributed OR scheduling) [S]; Roshanaei et al., EJOR 257(2):439–455 (2017) propagating LBBD for distributed OR scheduling, reported speed-ups of at least two orders of magnitude [S].
- ML for the inputs: Quantile Regression Forests feeding a MILP for rooms, sequences and times (OR Spectrum, 2025) [S]; predict-then-optimize for elective orthopaedics (retrospective study, PMC12422739, Sept 2025): lower overtime, **more idle time** than mean-duration schedules [S]; RL for intraday scheduling as a cooperative Markov game (Liu & Chen, arXiv 2025) [S].

### 2.4 Beyond the OR: beds, weekends, emergencies

- **Andam & Hashemi Doulabi**, arXiv 2602.15269 (2026) [S]: two-stage stochastic program with sample-average approximation; pooling ICU/ward beds across specialties; uncertainty in surgery duration and length of stay; full pooling reported to improve system performance "by up to 19.53%".
- **arXiv 2608.11018 (2026)** [S]: integrated master surgical schedule, surgeon/patient-to-block assignment, sequencing and weekday/weekend bed occupancy; listed as MILP + column-generation heuristic; supports block, open and modified-block policies.
- Rolling horizon and emergencies [S]: Kamran, Karimi, Dellaert & Demeulemeester (reserved slack, two-phase heuristics); Spratt & Kozan (arXiv 1808.10139); Andersen, Stidsen & Reinhardt (rolling and overlapping horizon).

### 2.5 Metaheuristics, CP-SAT and benchmarks

- Random-key algorithms for integrated OR scheduling (arXiv 2501.10243; Applied Soft Computing 2025) [S]; a systematic review of metaheuristics for operating-theatre planning (arXiv 2008.04970) [R].
- Healthcare Timetabling Competition 2024 (IHTC 2024): includes an OR/room assignment component; a public CP-SAT formulation exists (blog, Aug 2025) and a third-place hybrid MILP + CP + simulated-annealing entry (arXiv 2511.04685) [S].
- Marques, Captivo & Pato later used a genetic heuristic because the exact approach could not produce feasible solutions for high-dimensional instances [S].

### 2.6 Surveys (2020s)

- Al Amin, Baldacci & Kayvanfar, *Operational Research* 25 (2025), DOI 10.1007/s12351-024-00884-z: reviews 2000–2023, six groups by planning horizon, certainty vs uncertainty; flags uncertainty handling as open [S].
- Bibliometric analysis of OR scheduling research (PMC12358003) [S]; OR and surgical-team scheduling review, J. Project Management 2024 [S]; Zhu et al., J. Comb. Optim. (2019) [R].

## 3. What this means for our reproduction (my inferences, not claims of the sources)

1. **A stronger baseline exists.** If the Roshanaei & Naderi numbers hold, logic-based Benders beat the 2016 algorithm (0.88% vs 2.81%) on the same data. A fair next experiment is an LBBD or branch-and-check implementation on our generated instances, next to our CP-SAT monolithic baseline.
2. **Pricing is the scaling bottleneck for us.** At 300 cases and 10 days, 30 minutes gave only about 33 column-generation iterations. Two published remedies: ML ranking of pricing problems (Koutecká et al.) and labeling/dynamic-programming pricing instead of CP (Bargetto et al.; Xu et al.; Müller et al.). Neither is tested by us.
3. **Good initial columns help.** The 2026 follow-up gets them from RL + GA; our day-decomposition plan (day-level bound + per-day CP-SAT) served the same role and made the 300-case problem solvable. Worth comparing quality on the same instances.
4. **Our sensitivity finding has a published remedy.** Our Riverside analysis showed a 10% duration overrun makes urgent deadlines impossible. Distributionally robust chance constraints (Xu et al.; Wang et al.) are the closest published way to plan against that, but they cost optimality (the predict-then-optimize study saw more idle time).
5. **Real-world gaps the 2016 model ignores:** downstream beds and weekends, nurses and human resources, emergency insertion and live reoptimisation. All have recent papers (§2.1, 2.4).
6. **Benchmarks.** I found no evidence of a single standard public IORPS benchmark set (this was not searched exhaustively). IHTC 2024 is a recent public competition that overlaps partly.

## 4. Open questions and limits of this note

- All [S] numbers (0.88%, 3.84%, 19.53%, the up-to-30% ML speed-up) are as reported by search summaries; none was checked against full text.
- Whether Roshanaei & Naderi used the 2016 paper's exact instances, and with what time limits, is unknown.
- The 4.13% emergency-insertion figure is unattributed and should be ignored until the source is read.
- Coverage was one pass of about 15 searches (see below); 2025–2026 coverage was thin for some topics.
- Not searched yet: block scheduling and master surgery schedule literature after 2016; nurse/anaesthetist integration beyond the items above; Benders for the 2016 problem structure specifically; public instance repositories.

## 5. Papers to obtain (priority order)

1. Roshanaei & Naderi (2021), EJOR 293(1):65–78, DOI 10.1016/j.ejor.2020.12.004. Head-to-head with the 2016 paper.
2. Dolatkhah, Hashemi Doulabi, Rei & Gendreau, arXiv 2604.22977. Direct follow-up by the same group.
3. Bargetto, Garaix & Xie (2023), Computers & OR 152:106136, DOI 10.1016/j.cor.2022.106136 (HAL hal-04609352).
4. Koutecká, Šůcha, Hůla & Maenhout (2025), EJOR 320(2):328–342.
5. Akbarzadeh & Maenhout (2025), EJOR 322(2):448–466.
6. Xu, Yin & Wang, distributionally robust multi-period OR scheduling (need venue).
7. Müller, Moreno & Weiland (2026), Optimization Online, live OR reoptimization.
8. arXiv 2509.24806 (bilevel) and arXiv 2608.11018 (weekend beds).

## 6. Search log (2026-11-28)

Queries run (standard mode): integrated OR planning and scheduling + branch-and-price/CG/CP; the authors' names + topic; OR scheduling reviews 2023–24; stochastic/robust durations + branch-and-price; ML duration prediction and RL; Marques/Captivo/Pato; Koutecká ML pricing; Bargetto/Garaix/Xie; logic-based Benders for OR scheduling; CP-SAT OR scheduling; dynamic/emergency/rolling horizon; Xu/Yin/Wang DRO; Roshanaei & Naderi; live reoptimisation; the RL-CG, bilevel, weekend-bed and bed-pooling papers by title; ML for column generation.
