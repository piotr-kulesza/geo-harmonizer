// Shared categorical palette for "colour by dataset" — used by both the 3D
// landscape (Act 2) and the harmonize PCA view (Act 1) so datasets read the same
// colour across the whole app. Restrained, modern, reasonably colour-blind safe.
export const DATASET_COLORS = ['#4f9dd9', '#e0b64f', '#7ed08f', '#d94f6a', '#a17ee0', '#e08a4f']

export function datasetIndex(name, order) {
  const i = order.indexOf(name)
  return (i < 0 ? 0 : i) % DATASET_COLORS.length
}

// "#rrggbb" for a dataset given the reveal order (stable colour per dataset).
export function datasetHex(name, order) {
  return DATASET_COLORS[datasetIndex(name, order)]
}
