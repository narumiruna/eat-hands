# download images
URL=https://www.dropbox.com/s/n3gc2l644dxntrm/Hands.zip?dl=0
ZIP_FILE=./data/Hands.zip
mkdir -p ./data
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./data/
rm $ZIP_FILE

# download info
URL=https://www.dropbox.com/s/91dxij96vi5ve56/HandInfo.txt?dl=0
ZIP_FILE=./data/HandInfo.txt
wget -N $URL -O $ZIP_FILE
