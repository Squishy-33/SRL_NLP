# Data for the second NLP 2020 homework

## Warning
The data included in this package is copyrighted:
* DO NOT REDISTRIBUTE the data in any form.
* DO NOT UPLOAD the data on GitHub, personal webpage.
* DO NOT SHARE the data with anyone outside of the NLP 2020 course.
You MUST follow the above rules even after the homework deadline(s).

## Contents
* `train.json`: the train dataset.
* `dev.json`: the development/validation dataset.
* `test.json`: the test dataset.

## Data format
Each file is a JSON object organized as follows:
```json
{
    "sentence_id": {
        "words": ["word_0", "word_1", "...", "word_n"],
        "lemmas": ["lemma_0", "lemma_1", "...", "lemma_n"],
        "pos_tags": ["pos_tag_0", "pos_tag_1", "...", "pos_tag_n"],
        "dependency_heads": ["head_0", "head_1", "...", "head_n"],
        "dependency_relations": ["relation_0", "relation_1", "...", "relation_n"],
        "predicates": ["sense_0", "sense_1", "...", "sense_n"],
        "roles": {
            "predicate_index": ["role_0", "role_1", "...", "role_n"],
        }
    }
}
```
For example:
```json
{
    "1": {
        "words": [
            "Ms.",
            "Haag",
            "plays",
            "Elianti",
            "."
        ],
        "lemmas": [
            "ms.",
            "haag",
            "play",
            "elianti",
            "."
        ],
        "pos_tags": [
            "NNP",
            "NNP",
            "VBZ",
            "NNP",
            "."
        ],
        "dependency_heads": [
            "2",
            "3",
            "0",
            "3",
            "3"
        ],
        "dependency_relations": [
            "TITLE",
            "SBJ",
            "ROOT",
            "OBJ",
            "P"
        ],
        "predicates": [
            "_",
            "_",
            "PERFORM",
            "_",
            "_"
        ],
        "roles": {
            "2": [
                "_",
                "Agent",
                "_",
                "Theme",
                "_"
            ]
        },
    },
}
```