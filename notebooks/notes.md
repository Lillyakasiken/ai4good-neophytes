few shot experiment notes by a.s:
1. score only species that are in the training flight. the others cannot be found/scored.
2. my idea: we do a slight shift:
currently we have 6 species in the data. the original idea was that there exists a 7th species that only has 1 flight... my idea:
we basically train on all 5 species, then train on the "new" single flight of the 6th species (of which we have at the very least 2 flights). which allows us to train the model as if there was a new species while still have the ability to evaluate/test the model performance
Issues: 
 2a. In the five-species stage, drop every tile that contains the sixth species. If you mark those pixels as background, the model learns to suppress the plant you later want it to find.
 2b. Do not fine-tune on the new flight alone. Keep training on the old five as well, or it will forget them.
3. Explorer, Runs, Few-shot tab. One species is held out. Train on the other five plus one flight of it (joint, or a five-species base then a fine-tune that still replays the five). Test on its other flights, or on the fold's held-out sites. The plan diagram shows every flight before the run, then the epoch and species F1 while it runs. Click a flight to see its tiles on the map. Tiles of the species outside the chosen flight are dropped, not labelled background. 