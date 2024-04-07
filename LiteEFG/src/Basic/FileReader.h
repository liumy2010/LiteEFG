#ifndef FILEREADER_H_
#define FILEREADER_H_

#include <cstdio>
#include <cstdlib>
#include <fstream>

class FileReader {
public:
    static const int MAX_BUFFER_SIZE = 1000000;
    char buffer[MAX_BUFFER_SIZE + 5];

    char *cursor, *file_end;
    FILE *file;

    FileReader(const std::string& file_name_);

    char Read();

    void NextLine();

    void NextWord();

    char GetWord(std::string& word);
};

#endif