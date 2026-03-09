# Changelog

All notable changes to this project will be documented in this file.

## [0.0.10](https://github.com/jejjohnson/xrpatcher/compare/v0.0.9...v0.0.10) (2026-03-09)


### Features

* dynamic badges, add caching example, verify PyPI publish workflow ([e5d6cf2](https://github.com/jejjohnson/xrpatcher/commit/e5d6cf2b1c9d05375bf1fa70a2e17d52eee5b2bb))
* update badges, add caching example, and document publish workflow ([34328fe](https://github.com/jejjohnson/xrpatcher/commit/34328fee0ac1695b7f95baf23add9aa2c987adf8))


### Bug Fixes

* **examples:** warm cache before benchmark and use _patch variable name ([dfe1573](https://github.com/jejjohnson/xrpatcher/commit/dfe15737600c362fcebcfc7a307f9d84b4cc6f66))

## [0.0.9](https://github.com/jejjohnson/xrpatcher/compare/v0.0.8...v0.0.9) (2026-03-09)


### Features

* add GitHub Actions workflow to publish to PyPI on release ([6b3622c](https://github.com/jejjohnson/xrpatcher/commit/6b3622c844edce43c972e60f60c9040fbf94101f))
* add PyPI publish workflow triggered by release-please releases ([dfb65c8](https://github.com/jejjohnson/xrpatcher/commit/dfb65c856e76f5b562fc19e7b0f8295003d2a069))


### Documentation

* clarify trusted publisher setup wording in README ([3f09d8a](https://github.com/jejjohnson/xrpatcher/commit/3f09d8a0ff0a02d4e6b5db6323154ceba46f4c2e))

## [0.0.8](https://github.com/jejjohnson/xrpatcher/compare/v0.0.7...v0.0.8) (2026-03-09)


### Documentation

* add TorchGeo DataModule integration tutorial ([d293b31](https://github.com/jejjohnson/xrpatcher/commit/d293b3127863d8c5995f5b7010f639a6187e365c))
* add torchgeo datamodule tutorial example ([aa07845](https://github.com/jejjohnson/xrpatcher/commit/aa078455e83a4fd5a43f1d0e7fb83f0573d807be))
* fix torchgeo tutorial reconstruction mask ([f99bb4b](https://github.com/jejjohnson/xrpatcher/commit/f99bb4bd297c1bedbb2f0af0fd35f977d8ca66e6))
* fix torchgeo tutorial y-axis coordinates ([232d130](https://github.com/jejjohnson/xrpatcher/commit/232d1300bff0c5ae8c2566f8b3a297695dc97e0c))

## [0.0.7](https://github.com/jejjohnson/xrpatcher/compare/v0.0.6...v0.0.7) (2026-03-07)


### Features

* add in-memory caching for repeated patch access ([63f16f1](https://github.com/jejjohnson/xrpatcher/commit/63f16f1f89b5f8c4d6c544ba57f6dda8d6106e4f))
* add in-memory patch caching to XRDAPatcher ([0426752](https://github.com/jejjohnson/xrpatcher/commit/0426752da1dfb7a683d38e4758d3111686c3b519))


### Bug Fixes

* bypass cache for internal coordinate access ([3603746](https://github.com/jejjohnson/xrpatcher/commit/360374610d759606b6fa2017730b0ec35d4cca68))
* validate preload cache configuration ([325e3f1](https://github.com/jejjohnson/xrpatcher/commit/325e3f1cba082d54a1b86c457b6cf4bbcf0c4caf))


### Documentation

* align cache parameter docstrings ([64cece6](https://github.com/jejjohnson/xrpatcher/commit/64cece65e031f8a7d417db2a69ee886f81604826))
* clarify internal cache bypass helper ([5d69fea](https://github.com/jejjohnson/xrpatcher/commit/5d69fea8e4152b152c5f904e998f4d324368e239))
* clarify preload cache behavior ([ff9c596](https://github.com/jejjohnson/xrpatcher/commit/ff9c596397d37b48b076aa240ac6e045801bdb13))

## [0.0.6](https://github.com/jejjohnson/xrpatcher/compare/v0.0.5...v0.0.6) (2026-03-02)


### Bug Fixes

* address review comments on AGENTS.md and Makefile docs targets ([571d31e](https://github.com/jejjohnson/xrpatcher/commit/571d31e96b833e23d4afb12223fbf9fc850706c4))

## [0.0.5](https://github.com/jejjohnson/xrpatcher/compare/v0.0.4...v0.0.5) (2026-03-01)


### Bug Fixes

* align CI type-check/lint commands and fix type errors in base.py ([5048dc9](https://github.com/jejjohnson/xrpatcher/commit/5048dc9af7a070d33722333b72261e281152bf56))
* align CI type-check/lint commands and fix type errors in base.py ([5494989](https://github.com/jejjohnson/xrpatcher/commit/5494989c96a9dac48f618479d96a6451a9aeaa66))
* use indexers= keyword and order-preserving filter for coords_labels ([a5accc5](https://github.com/jejjohnson/xrpatcher/commit/a5accc5126db531fcb0cc49ba656256104e91b98))

## [0.0.4](https://github.com/jejjohnson/xrpatcher/compare/v0.0.3...v0.0.4) (2026-03-01)


### Bug Fixes

* fix docstring in get_coords ([715afc2](https://github.com/jejjohnson/xrpatcher/commit/715afc29e7205bb362092bbeb6eef4eafe0389c7))
* fix ty type errors in utils.py and base.py ([0e0a62d](https://github.com/jejjohnson/xrpatcher/commit/0e0a62dad39b0488d7322993d558187fdc5094fc))
* move ty python-version under [tool.ty.environment] ([7e8eb6b](https://github.com/jejjohnson/xrpatcher/commit/7e8eb6b3f53a471f752a1f370be5c5c398aff405))
* resolve ty typecheck CI failures ([31e0ee5](https://github.com/jejjohnson/xrpatcher/commit/31e0ee58847e334e3f413016d739dc293bb5d25d))

## [0.0.3](https://github.com/jejjohnson/xrpatcher/compare/v0.0.2...v0.0.3) (2026-03-01)


### Documentation

* expand Conventional Commits format in AGENTS.md to match CI validator ([271beb5](https://github.com/jejjohnson/xrpatcher/commit/271beb5ab56f2240a3f495d06f937b901445d90f))
* update Conventional Commits format in AGENTS.md with scope, breaking change, and lowercase requirement ([5348543](https://github.com/jejjohnson/xrpatcher/commit/53485431d2f549a7fcd9c7c63e152bf596f17b90))

## [0.0.2](https://github.com/jejjohnson/xrpatcher/compare/v0.0.1...v0.0.2) (2026-03-01)


### Bug Fixes

* remove PyPI publish from release workflow, use plain vX.X.X tags, reset to v0.0.1 ([9c32cf3](https://github.com/jejjohnson/xrpatcher/commit/9c32cf34cf6ba0ba25b7f49b68d98525c70ac649))
* remove PyPI publishing and fix release tag format to vX.X.X ([a8aad12](https://github.com/jejjohnson/xrpatcher/commit/a8aad125372a3216e1c519030b075a09e60ca758))
* **tests:** test bugs, add comprehensive test coverage, and add CI coverage infrastructure ([0db751a](https://github.com/jejjohnson/xrpatcher/commit/0db751ae618c0ffa574b64152279bbf5f9d420d5))

## [0.1.1](https://github.com/jejjohnson/xrpatcher/compare/xrpatcher-v0.1.0...xrpatcher-v0.1.1) (2026-03-01)


### Bug Fixes

* **tests:** test bugs, add comprehensive test coverage, and add CI coverage infrastructure ([0db751a](https://github.com/jejjohnson/xrpatcher/commit/0db751ae618c0ffa574b64152279bbf5f9d420d5))

## [0.1.0] - 2026-02-28

### Added

- Initial release of xrpatcher
- `XRDAPatcher` class for patching xarray DataArrays
- Support for arbitrary-dimension patching and reconstruction
- Overlap-aware reconstruction with optional weighting
