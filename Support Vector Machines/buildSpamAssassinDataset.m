function [X, y] = buildSpamAssassinDataset()

spamDir = 'C:\Knowledge\coursera\stanford-machine_learning\exercises\mlclass-ex6\spamassassin\data\spam';
nonspamDir = 'C:\Knowledge\coursera\stanford-machine_learning\exercises\mlclass-ex6\spamassassin\data\non-spam';

X = [];
y = [];

addpath(spamDir);
addpath(nonspamDir);
[spamEmailFiles, ~, ~] = readdir(spamDir);
y = [y; ones(rows(spamEmailFiles) - 2, 1)]; % augment y = 1s to the y vector, since this is a positive case. The minus 2 is required to ignore the root directories "." and ".."
for i = 1 : rows(spamEmailFiles)
	disp(i); 
	currentFile = spamEmailFiles{i};
	if strcmp(currentFile,".") || strcmp(currentFile,".."),
		continue;
	end
	
	fileContents = readFile(currentFile);
	word_indices = processEmail(fileContents);
	x = emailFeatures(word_indices);
	X = [X;x']; % augment new feature vector for the currently processed e-mail to the dataset
	disp(size(X));
end

disp("completed processing spam folder.");

[nonspamEmailFiles, ~, ~] = readdir(nonspamDir); % augment y = 0s to the y vector, since this is a negative (non-spam) case. The minus 2 is required to ignore the root directories "." and ".."
y = [y; zeros(rows(nonspamEmailFiles) - 2, 1)]; 
for j = 1 : rows(nonspamEmailFiles)
	disp(j);
	currentFile = nonspamEmailFiles{j};
	if strcmp(currentFile,".") || strcmp(currentFile,".."),
		continue;
	end
	
	fileContents = readFile(currentFile);
	word_indices = processEmail(fileContents);
	x = emailFeatures(word_indices);
	X = [X;x']; % augment new feature vector for the currently processed e-mail to the dataset
end;

end;