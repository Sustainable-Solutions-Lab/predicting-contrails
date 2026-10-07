# Literature concordance — does our corpus reproduce independent findings?

Our labels are CoCiP-family simulations (via the contrails.org pipeline),
so a first-order validation is whether the aggregate patterns in our
22.4M-flight 2021 corpus reproduce what independent groups — different
flight data, different met, different implementations, in some cases
different methods entirely — have published. Statistics below computed by
`concordance_stats.py` over all 2021 shards (net forcing 344 Mt CO2e-eq).

| Pattern | Our 2021 corpus | Published | Source |
|---|---|---|---|
| Forcing concentration | worst **2%** of flights → **72%** of net forcing; ~2.6% → 80%; worst 1% → 51% | **2%** of flights → **80%** of 2019 global energy forcing | Teoh et al. 2024 (ACP, global 2019–2021) |
| Regional concentration varies | (not directly computed per-region) | North Atlantic **12%** → 80%; Japan **2.2%** → 80% | Teoh et al. 2022 (ACP); Teoh et al. Japan study |
| Share of flights net-warming | **18%** warming, 8% cooling, 74% ≈ zero | ~**14%** of flights form a net-warming contrail | Teoh et al. 2024 |
| Night dominance | flights with route-mean darkness >50% (62% of traffic, COVID-year long-haul skew) contribute **96%** of net forcing; daytime flights net only +4% | night flights (25% of traffic) contribute **60–80%** of contrail forcing | Stuber et al. 2006 (Nature) |
| Winter dominance (NH mid-lat) | winter = **20%** of flights → **34%** of net forcing (mean 29.6 t/flight, 3.2× summer's 9.1 t) | winter = 22% of traffic → **~50%** of annual forcing | Stuber et al. 2006 |
| Regional structure | NAm 40% and Europe 22% of net forcing; Asia's per-km forcing (5.7 kg/km) lowest of major regions; Russia's highest (34.8) | net RF largest over Europe and USA; East Asia near global mean — fewer persistent contrails from lower cruise altitudes + subtropical (Hadley) dryness | Teoh et al. 2024 |
| Magnitude context | (we report EF as CO2e; not an RF estimate) | 2019 global contrail net RF 62.1 mW m⁻²; contrail-cirrus ERF 57 [17–98] mW m⁻² | Teoh et al. 2024; Lee et al. 2021 |

Reading: concentration, night-share, winter-share, warming-flight share,
and regional structure all land within the published envelope; where we
differ (night share above Stuber's 60–80%) the difference has a clear
mechanical explanation (our *net* accounting lets daytime warming and
cooling nearly cancel; their diurnal experiments perturbed traffic
fractions). Autumn ranking alongside winter in our 2021 numbers (33.4%
vs 33.6%) partly reflects 2021's traffic recovery profile — worth one
sentence in the paper.

Sources: Teoh et al. 2024, https://acp.copernicus.org/articles/24/6071/2024/ ·
Teoh et al. 2022, https://acp.copernicus.org/articles/22/10919/2022/ ·
Stuber et al. 2006, https://www.nature.com/articles/nature04877 ·
Lee et al. 2021 (aviation ERF assessment).
