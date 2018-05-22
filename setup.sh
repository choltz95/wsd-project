# get the GloVe package by running the following command
mkdir external_tools
cd external_tools
git clone https://github.com/stanfordnlp/GloVe 
make
cd ..

# download text8 data
CORPUS=../data/text8 
if [ ! -e $CORPUS ]; then
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip -d $DATADIR
rm text8.zip
fi

# download needed data for learning dictionary on window vectors 
cd data
wget https://www.dropbox.com/s/abrpkdvljjplbte/enwiki_vocab.txt
wget -c https://www.dropbox.com/s/0bs5oohqwhesc4a/enwiki_sq_vectors.bin
wget -c https://www.dropbox.com/s/puus99m7xgw75sj/poliblogs2008_para.mat
cd ..
