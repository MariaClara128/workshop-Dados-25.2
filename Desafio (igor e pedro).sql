CREATE TABLE Livros (
Nome VARCHAR(50) NOT NULL, 
Author VARCHAR(100) NOT NULL, 
CodeID INT NOT NULL, 
Ano_publicacao INT NOT NULL,
genero VARCHAR (50) NOT NULL,
PRIMARY KEY(CodeID));

CREATE TABLE Emprestimos (
Nome_cliente VARCHAR(100) NOT NULL,
CpfID VARCHAR(11) NOT NULL,
Data_emprestimo DATE NOT NULL,
CodeID INT NOT NULL,
PRIMARY KEY(CpfID),
FOREIGN KEY (CodeID) REFERENCES Livros(CodeID));

INSERT INTO Livros (Nome, Author, CodeID, Ano_publicacao, genero)
VALUES('As ondas do mar', 'jubas', 555444, 2007, 'suspense e romance'),
('Esse maldito coração', 'cachins', 192021, 2004, 'romance'),
('O infinito', 'lilly', 271631, 2024, 'romance'),
('A ultima chance', 'belly', 222211, 2025, 'romance e drama'),
('A hora da Estrela', 'Clarice Lispector', 233543, 1977, 'romance'),
('A casa mal assombrada', 'thomas hobb', 847847, 1943, 'terror'),
('Harry Potter', 'J.K. Rowling', 324232, 1997, 'fantasia'),
('Teto para Dois', 'Beth OLeary', 453655, 2005, 'romance'),
('O impossivel', 'agatha lins', 382611, 2002, 'acao'),
('As mil partes do meu coracao', 'Colleen Hoover', 564640, 2001, 'romance');

INSERT INTO Emprestimos (Nome_cliente, CpfID, Data_emprestimo, CodeID)
VALUES('Maria julia', '71745423542', '2024-03-06', 324232),
('Maria Clara', '71774845332', '2025-08-13', 222211),
('Amanda', '12134554321', '2025-07-24', 453655),
('Pedro', '43454223212', '2021-02-14', 847847),
('livia', '23421254312', '2023-01-08', 564640),
('Lucas', '74745434523', '2010-10-20', 382611),
('ludimilla', '71454376749', '2007-10-30', 233543),
('Igor', '71774554842', '2024-11-22', 555444),
('Corand', '72773475642', '2025-01-22', 192021), 
('belly', '12174543212', '2025-02-23', 271631);

UPDATE Emprestimos
SET Nome_cliente = 'ludimilla'
WHERE CodeID = 453655; 

SELECT Nome, Author, Ano_publicacao
FROM Livros
Where genero LIKE '%romance%';

SELECT COUNT(*) AS total_livros,
       MAX(Ano_publicacao) AS mais_recente,
       MIN(Ano_publicacao) AS mais_antigo
FROM Livros; 

SELECT genero, COUNT(*) AS total
FROM Livros
GROUP BY genero;
SELECT Nome_cliente, COUNT(*) AS total_emprestimos
FROM Emprestimos
GROUP BY Nome_cliente;

SELECT e.Nome_cliente, 
e.Data_emprestimo, 
l.Nome AS Livros, 
l.Author
FROM Emprestimos e 
JOIN Livros l ON e.CodeID = l.CodeID;

SELECT * FROM Livros;
SELECT * FROM Emprestimos;











