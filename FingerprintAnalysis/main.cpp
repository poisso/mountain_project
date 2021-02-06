#include "Image.h"

int main(){
    Image img("./../data/clean_finger.png");
    
    img.anisotropicLocalSimilarity(1.3, 50, 70, 150, 180);

    img.show();

    img.save("./../anisotropic.png");

    return 0;
}
