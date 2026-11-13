# Vendored front-end assets

Copied unmodified from npm packages (via cdn.jsdelivr.net) so the app loads nothing from the network.
The only edit: the trailing `sourceMappingURL` comment was removed from the Bootstrap files, as the maps are not shipped.

| File | Package and version | License |
|---|---|---|
| `bootstrap.min.css`, `bootstrap.bundle.min.js` | bootstrap 5.3.0 | MIT (`LICENSE-bootstrap.txt`) |
| `bootstrap-icons/` (css, woff, woff2) | bootstrap-icons 1.10.5 | MIT (`LICENSE-bootstrap-icons.txt`) |
| `fonts/montserrat-latin-*.woff2` | @fontsource/montserrat 5.3.0 (Montserrat) | SIL OFL 1.1 (`LICENSE-montserrat.txt`) |
| `fonts/poppins-latin-*.woff2` | @fontsource/poppins 5.3.0 (Poppins) | SIL OFL 1.1 (`LICENSE-poppins.txt`) |
| `fonts/lora-latin-*.woff2` | @fontsource/lora 5.3.0 (Lora) | SIL OFL 1.1 (`LICENSE-lora.txt`) |

`fonts.css` is ours: the `@font-face` rules for the Latin subsets listed above.
The OFL permits bundling and redistributing the fonts with software, provided the license text travels with them.
