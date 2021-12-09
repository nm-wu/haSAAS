# haSAAS - Machine-Assisted Human Grading of Short-Text Answers

This is the implementation of the algorithms presented in the paper 

> A. Meisl, G. Neumann: Towards Better Support for Machine-Assisted
> Human Grading of Short-Text Answers, Presented at HICSS 55: Hawai’i
> International Conference on System Sciences, January 4, 2022

## Preliminaries of the installation

To use the program, one has to install the tree tagger
as described in

     https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/

e.g. in directory ~/src/haSAAS.

Install for English and German assessemnt at least the following
dictionaries

     english-chunker.par.gz
     english.par.gz
     german-chunker.par.gz
     german.par.gz

and the tagging scripts and install-tagger.sh.
You should have now subdirectories "cmd", bin", and "lib".

Set the TAGDIR environment variable e.g. like:

    cd  ~/src/haSAAS
    export TAGDIR=`pwd`

Test the installation e.g. like:

     echo 'Das ist ein Test.' | cmd/tagger-chunker-german
     echo 'Hello world!' | cmd/tree-tagger-english

Under macOS, to run these commands will require setting permissions
to the binary via security settings.

## Python packages

The script "haSAAS.py" imports a couple of Python packages.
In case these are missing in your installation, you should be able to
"pip install" thse.

We developed and tested this program with Python 3.8.

## Test data

The available test data was provided originally by 
by Mohler et.al.

    https://www.aclweb.org/anthology/P11-1076.pdf
    https://aclanthology.org/P11-1076/

The .csv file has to contain the columns "id", "question", 
"student_answer", "score_avg".

The file can be loaded via:
   preprocess("resources/testdata/en/*.mohlercsv")

We include for convenience the test data, which contains
the data from several files from the link above merged
into a single .csv file.

## Run the script

To run the script, execute

    python haSAAS.py 

note that the first run will take a while, since it will analyze
162 assessments + 4546 submissions with the tree tagger. Later
runs will be much faster.

You might consider to turn on the "verbose" flag the see more detailed
analysis data.


Alexander Meisl    
Gustaf Neumann

