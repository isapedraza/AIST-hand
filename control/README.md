# Control

Runtime policies that turn model predictions into stable, safe commands.

This layer owns temporal confirmation, smoothing, command filtering, safety
clipping, top-k postural selection, and reset behavior. It should not capture
camera frames, define neural networks, or talk directly to robot middleware.

Current control code still lives near its original callers while the repository
layout is being stabilized. Future refactors should move those policies here
without changing the surrounding pipeline boundaries.
