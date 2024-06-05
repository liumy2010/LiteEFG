#include "Basic/FileReader.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

FileReader::FileReader(const std::string& file_name_) {
    file = fopen(file_name_.c_str(), "r");
    if(file == NULL) throw std::runtime_error("File " + file_name_ + " not found");
    cursor = file_end = NULL;
}

char FileReader::Read(){
    if(cursor == file_end){
        cursor = buffer;
        file_end = buffer + fread(buffer, 1, MAX_BUFFER_SIZE, file);
        if(cursor == file_end) return EOF; // End of file
    }
    return *cursor++;
}

void FileReader::NextLine(){
    char c;
    while(true){
        c = Read();
        if(c == '\n' || c == EOF) break;
    }
}

void FileReader::NextWord(){
    /*
        Move the cursor to start of the next word
        Example:
        "abcd abcd"   ----->   "abcd abcd"
          ^                          ^
    */
    char c;
    while(true){
        c = Read();
        if(c == ' ' || c== '\n' || c == EOF) break;
    }
    while(true){
        c = Read();
        if(c == EOF) break;
        if(c != ' ' && c!= '\n') {--cursor; break;}
    }
}

char FileReader::GetWord(std::string& word){
    /*
        Read a word from the file and save in variable word
        return the next character after the word
    */
    word = "";
    char c;
    while(true){ // move to the start of a word
        c = Read();
        if(c == EOF) {word=""; return EOF;}
        if(c != ' ' && c!= '\n') {--cursor; break;}
    }

    while(true){
        c = Read();
        if(c == ' ' || c== '\n' || c == EOF) break;
        word += c;
    }
    return c;
}