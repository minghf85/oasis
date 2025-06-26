folder_path="sub1"
# Read all filenames in the folder
for file in "$folder_path"/*; do
    # Extract the filename (excluding the extension)
    filename=$(basename "$file")
    topicname="${filename%.*}"
    echo "Processing topic: $topicname"
    # Generate paths for csv and db files
    # config_path = "${folder_path}/${topicname}.yaml"

    # Run the python script
    # python main.py --config "${folder_path}/${topicname}.yaml"
    python twitter_simulation.py --config_path "${folder_path}/${topicname}.yaml"
    # python visualization/result_ana.py --topic_name "$topicname"
    # cd examples/experiment/twitter_gpt_example
done
