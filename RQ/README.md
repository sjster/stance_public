The file job_submission.py pulls tweets corresponding to hashtags in the file passed through the --file flag. The output folder is where the downloaded tweets in JSON files are located. This is passed using the --output_folder flag. Additionally, this folder gets synced to Google Drive 'Twitter_rsettlag' (Need to parameterize this)

python job_submission.py --file hashtags.in --output_folder /home/vt/twitter_data/general_jobs/
