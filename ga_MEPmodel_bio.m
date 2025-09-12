% Biological model
%   Linear connectivity MNs->RC
%   Linear connectivity RC->MNs
%
% Example usage
%   >>subj       = 1;  % subject 1~10
%   >>withRC     = 1;  % 0: biological model without RC
%                      % 1: biological model with RC
%   >>AMPAweight = []; % []:free parameter range [0,1]
%                      % or a fixed value within [0,1]
%   >>reRun      = 0;  % 0: Load fitted result and plot simulated MEP.  
%                      % 1: Rerun model fitting. Backup previous fitted result.
%   >>ga_MEPmodel_bio(subj,withRC,AMPAweight,reRun);

function ga_MEPmodel_bio(subj,withRC,AMPAweight,reRun)

    root=fileparts(mfilename("fullpath"));
    addpath(fullfile(root,'GA','ga_toolbox'))
    addpath(fullfile(root,'GA','gradient_toolbox'))
    
    if nargin<2
        withRC=2;
    end
    if nargin<3
        AMPAweight=[]; % range [0,1]
    end
    if nargin<4
        reRun=0;
    end

    %----- model setting -----
    ref = config_model_bio(subj,withRC,AMPAweight);
    
    %----- run GA -----
    if exist([root,filesep,ref.resultname],'file') && ~reRun  
        fprintf('Use fitted result: \n%s\n',ref.resultname)
        tmp=load([root,filesep,ref.resultname]); 
        p_post=tmp.p_post;
    else
        p_post = run_ga(ref);
    end
 
    %----- show result -----
    plotOn=1;
    MEPmodel_bio(p_post,ref,plotOn);                                       
end


%==========================================================================
function [error,ref] = objective_function(p,ref)
 
    [~,ref] = MEPmodel_bio(p, ref); 
    error = ref.error; 
        
end
%==========================================================================
function p_post = run_ga(ref)
  
    myfunc   = @objective_function;
    op       = -1;   % -1: find minimum 
                     %  1: find maximum
    
    LR = ref.model.boundary(:,1);
    UR = ref.model.boundary(:,2); 
    nParams = length(LR);

    conf.UR = UR;
    conf.LR = LR;
    conf.op = op;
    conf.myfunc = myfunc;
    conf.y_goal = ref;
    
    conf.gLoop = 10;   % for gradient search
    conf.gL    = -12;
    conf.gU    = 12;
    conf.gT    = abs(conf.gU-conf.gL)+1;
    conf.gTol  = 0.01;
    
    %------------------------------------------------------------------
    N1  = 60;  % population size
    N2  = 100; % crossover, number of pairs to crossover 
    N3  = 100; % mutation, number of pairs to mutate 
    tg  = 1;%10;%5;   % total generations

    %rng(200)
    %warning('off')
    K  = 0; % history of [average cost, best cost]
    KP = 0; % history of [best solution]
    KS = 0; % history of [best cost]
    w  = 1; % counter
    j  = 1; % counter
    
    figure;
    %%%%%%%%%%collect previous solutions%%%%%%%%%%%%%
    root=fileparts(mfilename("fullpath"));
    tmpname=fullfile(root,'fitted_results','bio',sprintf('result_bio_s%d.mat',ref.subj));
    solution_ini=[];
    if exist(tmpname,"file")
        disp([tmpname ' found.'])
        tmp = load(tmpname);
        solution_ini = tmp.p_post;       
    end
    for AMPAw=0.2:0.1:0.8
        tmpname=fullfile(root,'fitted_results','bio','fixed_AMPAweight',sprintf('result_bio_s%d[%g].mat',ref.subj,AMPAw));
        if exist(tmpname,"file")
            disp([tmpname ' found.'])
            tmp = load(tmpname);
            if ~isempty(ref.model.AMPAweight)
                tmp.p_post(12)=ref.model.AMPAweight; % fix ampa weight
            end
            solution_ini = [solution_ini; tmp.p_post];  
        end
    end
    % rectify min max
    for i=1:nParams
        solution_ini(:,i)=clip(solution_ini(:,i),LR(i),UR(i));
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %-----initialization-----
    disp('======== Initialization ========')
    P = population(N1,nParams,LR,UR); % generate [60 x nParams] random solutions
    P = [P;solution_ini]; % add pre-selected solutions
    [E,R]= evaluation(P,myfunc,ref); % E: evaluation fitness,  R:residual, error
    [P,E,R] = selection_best(P,E,R,N1,op);
    R1 = R(:,1);
    disp('done');
    disp(['Minimum cost: ',num2str(E(1))])
    disp('================================')
    E_crit = E(1);
    
    %-----loop-----
    while 1
        disp('======= Gradient search ========')
        [Para_E_grd,E_grd,R_grd] = gradient_search(P(1,:),R1,conf,E_crit);
        % replace
        if op*E_grd > op*E(1)
            P(1,:) = Para_E_grd;
            E(1)   = E_grd;
            R(:,1) = R_grd;
        end
        disp('done')
              
        disp('======= single-parameter matation ========')
        P_ = mutation_single(P(1,:),LR,UR);  
    
        [E_,R_] = evaluation(P_,myfunc,ref);
        disp('done')
        
        disp('======= Gradient search ========')
        for i = 1:length(E_)
            fprintf('[%d/%d] cost: %f\n',i,length(E_),E_(i));
            [Para_E_grd(i,:),E_grd(i),R_grd(:,i)] = gradient_search(P_(i,:),R_(:,i),conf,E_crit);
        end
        % replace
        index = op*E_grd > op*E_;
        P_(index,:) = Para_E_grd(index,:);  % update gradient mutation
        E_(index) = E_grd(index);
        R_(:,index) = R_grd(:,index);
        P = [P;P_]; % [(60 + nParams) x nParams] solutions
        E = [E,E_]; % [1 x (60 + nParams)] cost
        R = [R,R_]; % [timepoints x (60 + nParams)] residual
        disp('done')
                           
        %%%%%%%%%%
        % add to show, delete later
        [~,E_show,~] = selection_best(P,E,R,1,op);
        disp(['best after gradient: ', num2str(E_show)])
        %%%%%%%%%%
        
        % GA
        disp('GA search...')
        P(end+1:end+N1,:) = mutationV(P(1:N1,:),0.1,0.9,LR,UR); % + N1 solutions
        P(end+1:end+2*N2,:) = crossover(P,N2);                  % + N2*2 solutions
        P(end+1:end+2*N3,:) = mutation(P,N3);                   % + N3*2 solutions
         
    
        [E_,~,~] = evaluation(P(N1+nParams+1:end,:),myfunc,ref);
        E = [E,E_]; % cost [1 x (N1+N2+N3)*2+nParams] 
        
        % selection
        [P,E] = selection_uniq(P,E,N1,N1,op,LR,UR);  % select N1 solutions
        %[P,E,~] = selection_best(P,E,p,op);
        [~,R1,~]= evaluation(P(1,:),myfunc,ref); % R1: residual of best solution
        disp('done')
               
        K(w,1) = sum(E)/N1;  % average cost (for plot)
        K(w,2) = E(1);       % best cost (for plot)
        
        KP(w,1:nParams) = P(1,1:nParams);  % save best
        KS(w) = E(1);           % save best
        E_crit = E(1);
        disp('========')
        disp(['current best Loss: ' num2str(KS(w))]);
        disp('========')
        gof = fitness_function(ref.y0(:),R1);
    
        disp('========')
        disp(['current best R2: ' num2str(gof)]);
        disp('========')
        %%%%%%%
        % add to show, delete later
        if E_show>E(1)
            disp('GA works')
            GA_counter(w) = 1;
        else
            disp("GA doesn't work")
            GA_counter(w) = 0;
        end
        %%%%%%%%%%
                        
        w = w+1; % update generation 
        j = j+1; % update generation counter
        
        % online plot
        [~,houtput]=myfunc(KP(end,:),ref);
        clf

        subplot(511)
        plot(K(:,2),'b.'); 
        hold on;
        plot(K(:,1),'r.'); 
        title('Blue - Best            Red - Average')
        xlabel('Generation')
        ylabel('Loss function')
        grid on
        set(gca, 'YScale', 'log')

        subplot(512)
        plot(E,'b.'); 
        xlabel('Cromosomes')
        ylabel('Loss function')
        grid on
        set(gca, 'YScale', 'log')

        subplot(513);
        plot(KP(end,:),'-ko'); 
        title('paramter')

        subplot(514)
        plot(ref.y0(:),'k','linewidth',1.5); hold on;
        plot(houtput.sim.simMEP2(:),'r','linewidth',1);
        title('target & best fit')
   
        subplot(515)
        plot(GA_counter,'b.'); 
        xlabel('Generations')
        gatoshow = length(GA_counter(GA_counter==1))/length(GA_counter);
        title(['0--not work,1--work, total succeful rate: ', num2str(gatoshow)]);
        drawnow;
                
        % stop: number of generations
        if j > tg
            break
        end
        
        % stop: good fit
        if KS(end)<0.01
            j = tg+1;
            break
        end
                
    end
    
    % get final result
    if op == -1
        [minimum idx] = min(KS);
        find_parameter = KP(idx,:);
        minimum
        
    end
    
    if op == 1
        [maximum idx] = max(KS);
        find_parameter = KP(idx,:);
        maximum
    end
     
    % make a copy of previous fitted result
    if exist(fullfile(root,ref.resultname),'file')
        filename_backup=[ref.resultname(1:end-4) sprintf('_backup-%s',datestr(now,'yyyy-mmdd-HHMM')) '.mat'];
        copyfile(fullfile(root,ref.resultname),...
                 fullfile(root,filename_backup));
    end
    
    % save fitted result
    p_post = KP(end,:);
    [~,ref] = MEPmodel_bio(p_post,ref,0); % update ref
    save(fullfile(root,ref.resultname),'p_post','KP','ref','P','KS')
    disp('fitted result saved:')
    fprintf('%s\n',ref.resultname)

end
    