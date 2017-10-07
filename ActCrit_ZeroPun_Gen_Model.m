% Code: Harrison Satcher
% PI: Adam Morris
% Cushman Lab
% Model Free Punishments
% Created: 9/12/17
% Last modified: 10/6/17
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A generative model that simulate data in line with our proposed
% Actor-Critic Model Free learning without passing back punishments.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

numRounds = 1000; % Determined by number of trials
numSubj = 100;
genRewardTable = zeros(numRounds,6); % To generate nonstationary with reflection
rewardTable = zeros(numRounds,4); % True rewards table
maxReward = 4;
minReward = -4;
RewardStDev = 1.2;
resultsMatrix = zeros(numRounds, 7); % [terminal, reward, optimal terminal, optimal reward]
terminal = 0;
gamma = .9;
lambda = .95; % ask Adam
alpha = .3;
beta = .3;
tau = .8;
numStay = zeros(1,3);
numGo = zeros(1,3);

% Set rewards, altered from code provided by Adam Morris: 
% https://github.com/adammmorris/cached_values/blob/master/Simulations/env/createEnv.m
% Random Gaussian walk with boundary reflection
genRewardTable(1,:) = round(normrnd(0,RewardStDev,1,6));
for thisRound = 2:numRounds
     re = genRewardTable(thisRound-1, :) + round(normrnd(0,RewardStDev,1,6));
     re(re > maxReward) = 2 .* maxReward - re(re > maxReward);
     re(re < minReward) = 2 .* minReward - re(re < minReward);
     genRewardTable(thisRound,:) = re;
end

for i=1:numSubj
    % Normalizing symemetric rewards so beta reflects range 0 to 10, g. prior
    % rewardTable = (rewardTable + ((maxReward - minReward) ./ 2)) ./ (maxReward - minReward);
    rewardTable(:,5:10) = genRewardTable; %collapse reward tables
%     rewardTable(rewardTable(:,:) < -1) = -3;
%     rewardTable(rewardTable(:,:) > 1) = 3;
    rewardTable(rewardTable(:,:) > -2 & rewardTable(:,:) < 2) = 0;
    
    
    
    [maxReward, maxState] = max(rewardTable,[],2); % find optimal rewards
    
    currentState = 1;
    actorTable = zeros(4,2); % p(s,a) policy preferences
    criticTable = zeros(1,10); % V(s) state values
    traceStateTable = zeros(1,10); % eligibility traces for state values
    traceActionTable = zeros(4,2); % eligibility traces for action values

    for j=1:numRounds
        while ~terminal
            % Action selection; Gibbs softmax utilizing action preference
            normFactor = max(actorTable(currentState,:));
            probDist = exp((actorTable(currentState,:)- normFactor) .* tau) ./ sum(exp((actorTable(currentState,:) - normFactor).* tau)); 
            currentAction = randsample(1:2, 1, true, probDist); 
            if currentState == 1
                firstAction = currentAction;
                if (randsample(1:2,1,true, [20 80]) == 2)
                    nextState = currentState + currentAction;
                else
                    nextState = 4;
                end
            else
                nextState = (currentState.*2)+currentAction;
            end
            reward = rewardTable(j, nextState);
            % TD error
            del = reward + gamma .* criticTable(nextState) - criticTable(currentState);
            
            if reward > -1
                % eligibility trace update
                traceStateTable(currentState) = traceStateTable(currentState) + 1;
                traceActionTable(currentState, currentAction) = traceActionTable(currentState, currentAction) + 1;
                % critic updates self
                criticTable = criticTable + alpha .* del .* traceStateTable;
            else
                % eligibility trace update
                traceStateTable = zeros(1,10); % effectively set lambda to 0
                traceActionTable = zeros(4,2); % effectively set lambda to 0
                traceStateTable(currentState) = traceStateTable(currentState) + 1;
                traceActionTable(currentState, currentAction) = traceActionTable(currentState, currentAction) + 1;
                % critic updates self
                criticTable = criticTable + 0 .* del .* traceStateTable; 
            end
            % critic updates actor
            actorTable = actorTable + beta .* del .* traceActionTable; 
            traceStateTable = traceStateTable .* gamma .* lambda;
            traceActionTable = traceActionTable .* gamma .* lambda;
            currentState = nextState;
            if currentState > 4
                terminal = 1;
            end
        end
        resultsMatrix(j, :) = [firstAction, currentState, reward, del, criticTable(1), actorTable(1,1), actorTable(1,2)]; %log results; first action taken, T state, reward received
        % reset the task
        terminal = 0; 
        currentState = 1;
        traceStateTable = zeros(1,10);
        traceActionTable = zeros(4,2);
    end
    meanResults(1,i) = mean(resultsMatrix(:, 4));
    
    % If the round is sent to the probabilistic space, and the next round
    % attempts the same action again, mark as stay, otherwise goes.
    % numStay = [numStayNegative, numStayZero, numStayPositive]
    % Better way to do this comuptationally? 
    for k=1:numRounds-1
        if (resultsMatrix(k, 2) > 8)
            if (resultsMatrix(k+1,1) == resultsMatrix(k,1))
                if (resultsMatrix(k, 3) < 0)
                    numStay(1) = numStay(1) + 1;
                elseif (resultsMatrix(k, 3) > 0)
                    numStay(3) = numStay(3) + 1;
                else
                    numStay(2) = numStay(2) + 1;
                end
            elseif (resultsMatrix(k, 3) < 0)
                numGo(1) = numGo(1) + 1;
            elseif (resultsMatrix(k, 3) > 0)
                numGo(3) = numGo(3) + 1;
            else
                numGo(2) = numGo(2) + 1;
            end
        end
    end
% % %     csvwrite(strcat('Data_Model_Agents/', 'modelagent', string(i), '.csv'), ...
% % %             resultsMatrix);
end
numTot = numGo + numStay;
bar(numStay./numTot);

    
