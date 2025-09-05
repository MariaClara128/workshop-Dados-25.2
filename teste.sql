CREATE TABLE Aluno (
    Matricula INT PRIMARY  KEY  AUTO_INCREMENT,
    Nome VARCHAR (100) NOT NULL,
    CPF CHAR (11) UNIQUE NOT NULL
);
CREATE TABLE Aluno turma (
    Id_aluno INT,
    Id turma INT,
    PRIMARY KEY (id_aluno, id_turma),
    FOREIGN KEY (id_aluno) REFERENCS Aluno (id_aluno),
    FOREIGN KEY (id_turma) REFERENCES Turma(id_turma)
);
CREATE TABLE  Professor Turma (
    Id_professor INT,    
    Id_turma INT,
    PRIMARY KEY (id_professor, id_turma),
    FOREIGN KEY (id_professor) REFERENCES Professor (id_professor),
    FOREIGN KEY (id_turma) REFERENCES Turma (id_turma),
);
CREATE TABLE Turma(
   Periodo INT PRIMARY KEY.
   Turno VARCHAR (20) NOT NULL,
   Curso VARCHAR (50) NOT NULL,
);
CREATE TABLE Professor (
    ia professor INT PRIMARY KEY
    AUTO_INCREMENT,
    nome VARCHAR(100) NOT NULL,
    titulacao VARCHAR(50),
    cpf CHAR(11) UNIQUE NOT NULL,
);
