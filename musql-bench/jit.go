package main

import (
	"os"

	musqlengine "github.com/samyfodil/musql/engine"
)

// MUSQL_JIT=0 turns the JIT off, as in musql's own harness.
func init() {
	if os.Getenv("MUSQL_JIT") == "0" {
		musqlengine.Configure(musqlengine.WithoutJIT())
	}
}
