for file in *.html; do
  if [[ "$file" == *"delta_0.05_"* ]];then
    echo $file
    mv "$file" "${file/delta_0.05_/}" # delta_0.05_ was found
  fi
done


# for file in *.csv
# do
#   mv "$file" "${file/delta_0.05_/}"
# done