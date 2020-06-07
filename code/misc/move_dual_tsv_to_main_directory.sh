for fil in $(find ../../*block -name "*dual*.tsv"); do cp ${fil} ${fil/block/}; done
