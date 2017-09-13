% Code: Harrison Satcher
% PI: Adam Morris
% Cushman Lab
% Model Free Punishments
% Created: 9/12/17
% Last modified: 9/12/17
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A generative model that simulate data in line with our proposed
% Actor-Critic Model Free learning without passing back punishments.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

numRounds = 1000; % Determined by number of trials
numSubj = 1;
actorTable = [0,0;0,0;0,0;0,0]; % p(s,a) policy preferences
criticTable = [0,0,0,0,0,0,0,0,0,0]; % V(s) state values
genRewardTable = zeros(numRounds,4); % To generate nonstationary with reflection
rewardTable = zeros(numRounds,4); % True rewards table
maxReward = 5;
minReward = -5;
RewardStDev = 1.5;
currentState = 1;
resultsMatrix = zeros(numRounds, 4); % [terminal, reward, optimal terminal, optimal reward]
terminal = 0;
gamma = .9;
alpha = .5;
beta = 3;

% Write up Actor-Critic with critic punishment learning rate low
% Does it show our effect on this task? 
% 
%%% Short-term task: equal 0, -
%%% Long-term task: higher 0, lower -


for i=1:numSubj
    
% % %     alpha = rand; 
% % %     beta = randi([0,10]);

    % Set rewards, altered from code provided by Adam Morris: 
    % https://github.com/adammmorris/cached_values/blob/master/Simulations/env/createEnv.m
    % Random Gaussian walk with boundary reflection
	genRewardTable(1,:) = round(normrnd(0,RewardStDev,1,4));
    for thisRound = 2:numRounds
         re = genRewardTable(thisRound-1, :) + round(normrnd(0,RewardStDev,1,4));
         re(re > maxReward) = 2 .* maxReward - re(re > maxReward);
         re(re < minReward) = 2 .* minReward - re(re < minReward);
         genRewardTable(thisRound,:) = re;
    end
    % Normalizing symemetric rewards so beta reflects range 0 to 10, g. prior
    % rewardTable = (rewardTable + ((maxReward - minReward) ./ 2)) ./ (maxReward - minReward);
    rewardTable(:,5:8) = genRewardTable; %collapse reward tables
    rewardTable(:,9:10) = -5; % add punishment state
    [maxReward, maxState] = max(rewardTable,[],2); % find optimal rewards

    for j=1:numRounds
        while ~terminal
            % Action selection; Gibbs softmax utilizing action preference
            probDist = exp(actorTable(currentState,:)) ./ sum(exp(actorTable(currentState,:))); 
            currentAction = randsample(1:2, 1, true, probDist); 
            if currentState == 1
                nextState = currentState + currentAction;
            else
                nextState = (currentState.*2)+currentAction;
            end
            reward = rewardTable(j, nextState);
            % TD error
            if reward > 0 % are punishments passed back
                del = reward + gamma .* criticTable(nextState) - criticTable(currentState);
            else
                del = gamma .* criticTable(nextState) - criticTable(currentState);
            end
            % critic updates actor
            actorTable(currentState, currentAction) = actorTable(currentState, currentAction) + beta .* del;
            % critic updates self
            criticTable(currentState) = criticTable(currentState) + alpha .* del;
            currentState = nextState;
            if currentState > 4
                terminal = 1;
            end
        end
        resultsMatrix(j, :) = [currentState, reward, maxState(j), maxReward(j)]; %log results
        % reset the task
        terminal = 0; 
        currentState = 1;
    end
    plot(1:numRounds,resultsMatrix(:,2))
    hold on
    plot(1:numRounds,resultsMatrix(:,4))
    hold off
% % %     csvwrite(strcat('Data_Model_Agents/', 'modelagent', string(i), '.csv'), ...
% % %             resultsMatrix);
end


    
