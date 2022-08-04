## About

This repo contains some of the source code for the work 'Stance detection from weakly-supervised text'. This work attempts to learn projections from short documentsthrough language model embeddings. 

The folder structure is as follows:

```
.
├── App    
│   ├── README
│   └── vis.py
├── Elmo_pytorch.py            <--------- Pytorch code with Elmo embeddings for stance extraction 
├── README.md
├── RQ               
│   ├── README.md
│   ├── job.py
│   ├── job_submission.py
│   └── logging_mod.py
├── app.png
└── attention_pytorch_opt.py   <-------- Pytorch code for stance with hyperparameter optimization with comet (No Elmo)
```
