# Data Folder

processed contains the working snapshots used by this project. See [data notes](../docs/data_notes.md) for source references, dimensions, field interpretation, and changes to the public support copy.

The two Telco CSVs are supplied snapshots. The support CSV is reduced to remove names, emails, ticket descriptions, and resolution free text. No support records are joined to Telco records.

Optional source CSVs can be placed in data/raw with the filenames documented in the data notes. Generated data is written under outputs and does not overwrite the supplied snapshots automatically.
