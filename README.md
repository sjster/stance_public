## About

This repo contains some of the source code for the work 'Stance detection from weakly-supervised text'. This work attempts to learn projections from short documentsthrough language model embeddings. 

The folder structure is as follows:

.
├── App                       <------------------------ Web app in Plot.ly dash
│   ├── README
│   └── vis.py
├── Elmo_pytorch.py           <------------------------ Pytorch code using Elmo
├── README.md
├── RQ                        <------------------------ Tweet downloader (Updated and generic tweet downloader repo at https://github.com/sjster/Tweepy_streams_S3)
│   ├── README.md
│   ├── job.py
│   ├── job_submission.py
│   └── logging_mod.py
├── app.png
└── attention_pytorch_opt.py  <-------------------- Pytorch code that use comet for hyperparameter optimization (No Elmo)
