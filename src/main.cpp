#include <string>

using namespace std;


void parseProgramParameters(int argc, char* argv[]){
	if (argc<2){ // not enough arguments
		exit(0);
	}
	for (int i = 1; i < argc; i++) {
	}
}

int main(int argc, char *argv[]) {
	fprintf(stdout, "\n## PROGRAM PARAMETERS \n");
	parseProgramParameters(argc, argv);
}