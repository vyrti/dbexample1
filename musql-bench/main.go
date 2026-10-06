// Command musql-bench times musql on the data and bind values written by
// `sinew-bench prepare`, with the same procedure sinew-bench uses for the other
// engines, and prints TSV for `sinew-bench run --musql`.
//
// musql is measured the way its README's table measures it: the engine-direct
// API (ReadOnlyPager.QueryArgs, plan-cached) on a VACUUMed file, plus the
// database/sql driver for reference. MUSQL_JIT=0 turns the JIT off.
package main

import (
	"bufio"
	"database/sql"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/samyfodil/musql/convert/sqlite"
	"github.com/samyfodil/musql/driver"
	musqlengine "github.com/samyfodil/musql/engine"
)

const nCheck = 40

type workload struct {
	name, sql string
	args      [][]int64
}

func loadWorkloads(path string) ([]workload, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []workload
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		parts := strings.SplitN(sc.Text(), "\t", 3)
		switch parts[0] {
		case "W":
			out = append(out, workload{name: parts[1], sql: parts[2]})
		case "A":
			var a []int64
			if parts[1] != "" {
				for _, s := range strings.Split(parts[1], ",") {
					v, err := strconv.ParseInt(s, 10, 64)
					if err != nil {
						return nil, err
					}
					a = append(a, v)
				}
			}
			w := &out[len(out)-1]
			w.args = append(w.args, a)
		}
	}
	return out, sc.Err()
}

// timeIt mirrors sinew-bench's time_it: 3 warm-up calls, iterations sized from
// one timed call to take ~target, reps repetitions, median of per-call means.
func timeIt(nargs int, target time.Duration, reps int, run func(i int) error) (median, min time.Duration, err error) {
	for i := 0; i < 3; i++ {
		if err = run(i % nargs); err != nil {
			return
		}
	}
	t := time.Now()
	if err = run(3 % nargs); err != nil {
		return
	}
	one := time.Since(t)
	n := 3
	if one < target {
		n = int(target / max(one, 1))
		n = max(40, min2(n, 2_000_000))
	}
	means := make([]time.Duration, 0, reps)
	for r := 0; r < reps; r++ {
		t := time.Now()
		for i := 0; i < n; i++ {
			if err = run(i % nargs); err != nil {
				return
			}
		}
		means = append(means, time.Since(t)/time.Duration(n))
	}
	sort.Slice(means, func(a, b int) bool { return means[a] < means[b] })
	return means[len(means)/2], means[0], nil
}

func min2(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func toVals(a []int64) []musqlengine.Value {
	out := make([]musqlengine.Value, len(a))
	for i, x := range a {
		out[i] = musqlengine.Value{Typ: musqlengine.Int, I: x}
	}
	return out
}

func toAny(a []int64) []any {
	out := make([]any, len(a))
	for i, x := range a {
		out[i] = x
	}
	return out
}

// renderDirect is compat-harness's renderDirectInts.
func renderDirect(rp *musqlengine.ReadOnlyPager, q string, args []int64) (string, error) {
	_, rows, err := rp.QueryArgs(q, toVals(args))
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	for _, r := range rows {
		for _, v := range r {
			switch v.Typ {
			case musqlengine.Null:
				sb.WriteString("N:|")
			case musqlengine.Int:
				fmt.Fprintf(&sb, "i:%d|", v.I)
			case musqlengine.Float:
				fmt.Fprintf(&sb, "f:%v|", v.F)
			case musqlengine.Text, musqlengine.Blob:
				fmt.Fprintf(&sb, "t:%s|", string(v.S))
			}
		}
		sb.WriteString(";")
	}
	return sb.String(), nil
}

// queryDriver runs through database/sql and scans every cell, as the
// harness's renderDriverInts does (minus the rendering).
func queryDriver(db *sql.DB, q string, args []int64) error {
	rows, err := db.Query(q, toAny(args)...)
	if err != nil {
		return err
	}
	defer rows.Close()
	cols, _ := rows.Columns()
	cells := make([]any, len(cols))
	ptrs := make([]any, len(cols))
	for i := range cells {
		ptrs[i] = &cells[i]
	}
	for rows.Next() {
		if err := rows.Scan(ptrs...); err != nil {
			return err
		}
	}
	return rows.Err()
}

func main() {
	dir := flag.String("dir", "../data", "directory written by sinew-bench prepare")
	targetMs := flag.Int("target-ms", 200, "time per repetition")
	reps := flag.Int("reps", 5, "repetitions")
	flag.Parse()

	src := filepath.Join(*dir, "bench.db")
	dst := filepath.Join(*dir, "musql.musq")
	os.Remove(dst)
	t := time.Now()
	if err := sqlite.Import(src, dst, sqlite.ImportOptions{}); err != nil {
		fmt.Fprintln(os.Stderr, "import:", err)
		os.Exit(1)
	}
	db, err := sql.Open(driver.DriverName, dst)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	db.SetMaxOpenConns(1)
	// Every row into segments, none in the delta: the format at rest, as the
	// harness benchmarks it.
	if _, err := db.Exec("VACUUM"); err != nil {
		fmt.Fprintln(os.Stderr, "VACUUM:", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "musql: imported + VACUUM in %v, JIT=%v\n", time.Since(t).Round(time.Millisecond), musqlengine.JITEnabled())
	rp, err := musqlengine.Open(dst)
	if err != nil {
		fmt.Fprintln(os.Stderr, "engine open:", err)
		os.Exit(1)
	}
	defer rp.Close()

	wls, err := loadWorkloads(filepath.Join(*dir, "workloads.tsv"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "workloads:", err)
		os.Exit(1)
	}
	label := "musql"
	if !musqlengine.JITEnabled() {
		label = "musql (no JIT)"
	}
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()
	fmt.Fprintf(out, "L\t%s\n", label)
	target := time.Duration(*targetMs) * time.Millisecond
	for _, w := range wls {
		fmt.Fprintf(os.Stderr, "%-34s", w.name)
		renders := make([]string, 0, nCheck)
		var werr error
		for i := 0; i < nCheck && i < len(w.args); i++ {
			r, err := renderDirect(rp, w.sql, w.args[i])
			if err != nil {
				werr = err
				break
			}
			renders = append(renders, r)
		}
		var dMed, dMin, vMed time.Duration
		if werr == nil {
			dMed, dMin, werr = timeIt(len(w.args), target, *reps, func(i int) error {
				_, _, err := rp.QueryArgs(w.sql, toVals(w.args[i]))
				return err
			})
		}
		if werr == nil {
			vMed, _, _ = timeIt(len(w.args), target, *reps, func(i int) error {
				return queryDriver(db, w.sql, w.args[i])
			})
		}
		errText := ""
		if werr != nil {
			errText = strings.ReplaceAll(werr.Error(), "\t", " ")
		}
		fmt.Fprintf(out, "W\t%s\t%d\t%d\t%d\t%s\n", w.name, dMed.Nanoseconds(), dMin.Nanoseconds(), vMed.Nanoseconds(), errText)
		for _, r := range renders {
			fmt.Fprintf(out, "C\t%s\n", r)
		}
		fmt.Fprintf(os.Stderr, " direct %v  driver %v %s\n", dMed, vMed, errText)
	}
}
