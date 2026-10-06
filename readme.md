# sinew (Rust) vs musql (Go)

100,000 rows on Apple M1. Median per-query time over five repetitions of approximately 200 ms each, using identical data and bind values. Sinew uses its SQL-text API with assembly kernels; musql uses its direct engine API with JIT enabled on a VACUUMed database.

| Query | sinew (Rust) | musql (Go) | sinew speedup |
| --- | ---: | ---: | ---: |
| Filtered count, one predicate | 8.8 µs | 40.1 µs | **4.6×** |
| Filtered count, two predicates | 11.9 µs | 41.2 µs | **3.5×** |
| Rowid lookup | 62 ns | 10.2 µs | **164×** |
| Secondary-index equality | 78 ns | 10.9 µs | **140×** |
| Indexed equi-join | 165 ns | 13.2 µs | **80.0×** |
| Sum over a filter | 8.6 µs | 458.3 µs | **53.6×** |
| Grouped aggregate | 80.9 µs | 8.73 ms | **108×** |
| ORDER BY v DESC LIMIT 20 | 2.6 µs | 1.37 ms | **519×** |
| Whole-table count | 212 ns | 11.0 µs | **51.7×** |
| min/max whole table | 635 ns | 363.9 µs | **573×** |
| Grouped min/max | 160.1 µs | 15.68 ms | **98.0×** |
| BETWEEN range | 16.9 µs | 15.89 ms | **941×** |
| Paginate OFFSET | 139.6 µs | 8.82 ms | **63.2×** |
| Join projecting rows | 155 ns | 12.9 µs | **83.4×** |

Speedups are calculated from unrounded measurements.
