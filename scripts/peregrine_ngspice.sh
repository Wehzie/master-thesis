module purge
module load Bison
module load flex
module load X11
module load fontconfig
module load freetype
module load Autoconf
module load Automake
module load libtool
module load libreadline


wget https://sourceforge.net/projects/ngspice/files/ng-spice-rework/39/ngspice-39.tar.gz/download
tar -zxvf ngspice-39.tar.gz
cd ngspice-39
./configure --enable-xspice --enable-osdi --disable-debug --with-readline=yes
make clean
make
sudo make install