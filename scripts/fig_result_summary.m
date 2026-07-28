parentDir = fileparts(pwd); 
addpath(parentDir);

%---------fig_result_summary.png----------
score=zeros(10,3); %[Pheno,Bio,BioNoRC];
score2=zeros(10,3); %[Pheno,Bio,BioNoRC]; % gof of IO curve
for subj=1:10
 disp(subj)
 tmp=load(fullfile(parentDir,'fitted_results','pheno',sprintf('result_pheno_s%d.mat',subj)));
 [~,ref]=MEPmodel_pheno(tmp.p_post,tmp.ref,0);
 score(subj,1)=ref.R2;
 score2(subj,1)=ref.gof;
  tmp=load(fullfile(parentDir,'fitted_results','bio',sprintf('result_bio_s%d.mat',subj)));
 [~,ref]=MEPmodel_bio(tmp.p_post,tmp.ref,0);
 score(subj,2)=ref.R2;
 score2(subj,2)=ref.gof;
  tmp=load(fullfile(parentDir,'fitted_results','bioNoRC',sprintf('result_bioNoRC_s%d.mat',subj)));
 [~,ref]=MEPmodel_bio(tmp.p_post,tmp.ref,0);
 score(subj,3)=ref.R2;
 score2(subj,3)=ref.gof;
end
%% --------------------------------------------
figure;
markerColor=[128 128 128]/255;
boxColor=[0,139,139]/255;
width=22;
height=9;
set(gcf,'units','centimeters','position',[2 2 width height])

subplot(2,8,[1 9])
x=repelem(1:3,10)'; y=score(:);
boxchart(x,y);hold on;
xticklabels(['Pheno.','RC+','RC-'])
plot(1:3,score','-o','Color',markerColor);
ylabel('Waveform R^2');
box off;ylim([0.5 1])
set(gca,'xticklabel',{'Pheno.','RC+','RC-'},'fontname','calibri')

subplot(242)
scatter(score(:,2),score(:,1),[],markerColor); hold on;
plot([0;1],[0;1],'k');
xlim([0.8 1]);ylim([0.8 1]);
ylabel('Pheno.'); xlabel('(RC+) model');
for i=1:10
    text(score(i,2)+0.005,score(i,1),sprintf('%d',i),'HorizontalAlignment','left','FontSize',8)
end
box off;axis square;


subplot(246)
scatter(score(:,2),score(:,3),[],markerColor); hold on;
plot([0;1],[0;1],'k');
xlim([0.8 1]);ylim([0.5 1]);
ylabel('(RC-) model'); xlabel('Bio.');
for i=1:10
    text(score(i,2)+0.005,score(i,3),sprintf('%d',i),'HorizontalAlignment','left','FontSize',8)
end
box off;axis square;


subplot(2,8,[5 13])
x=repelem(1:3,10)'; y=score2(:);
boxchart(x,y);hold on;
xticklabels(['Pheno.','RC+','RC-'])
plot(1:3,score2','-o','Color',markerColor);
ylabel('IO R^2');
box off;ylim([0.5 1])
set(gca,'xticklabel',{'Pheno.','RC+','RC-'},'fontname','calibri')

subplot(244)
scatter(score2(:,2),score2(:,1),[],markerColor); hold on;
plot([0;1],[0;1],'k');
xlim([0.8 1]);ylim([0.8 1]);
ylabel('Pheno.'); xlabel('(RC+) model');
for i=1:10
    text(score2(i,2)+0.005,score2(i,1),sprintf('%d',i),'HorizontalAlignment','left','FontSize',8)
end
box off;axis square;


subplot(248)
scatter(score2(:,2),score2(:,3),[],markerColor); hold on;
plot([0;1],[0;1],'k');
xlim([0.8 1]);ylim([0.5 1]);
ylabel('(RC-) model'); xlabel('(RC+) model');
for i=1:10
    text(score2(i,2)+0.005,score2(i,3),sprintf('%d',i),'HorizontalAlignment','left','FontSize',8)
end
box off;axis square;
saveas(gcf,['figures' filesep 'fig_result_summary.svg'])

%%
disp('waveform R^2 [Pheno,RC+,RC-]')
disp('--------------------------------')
disp(mean(score,1))

disp('IO curve R^2 [Pheno,RC+,RC-]')
disp('--------------------------------')
disp(mean(score2,1))

[h,p]=ttest(score(:,1),score(:,2))
[h,p]=ttest(score2(:,2),score2(:,3))

%% --------------------------------------------
figure;
markerColor=[128 128 128]/255;
boxColor=[0,139,139]/255;
width=15;
height=10;
set(gcf,'units','centimeters','position',[2 2 width height])

%----------waveform R2-------------
subplot(131)
x=repelem(1:3,10)'; y=score(:);
boxchart(x,y);hold on;
xticklabels(['Pheno.','RC+','RC-'])
plot(1:3,score','-o','Color',markerColor);
ylabel('Waveform R^2');
box off;ylim([0.5 1])
set(gca,'xticklabel',{'Pheno.','RC+','RC-'},'fontname','calibri')
%----------IO R2-------------
subplot(132)
x=repelem(1:3,10)'; y=score2(:);
boxchart(x,y);hold on;
xticklabels(['Pheno.','RC+','RC-'])
plot(1:3,score2','-o','Color',markerColor);
ylabel('IO R^2');
box off;ylim([0.5 1])
set(gca,'xticklabel',{'Pheno.','RC+','RC-'},'fontname','calibri')
%-------------diff---------------
subplot(133)
x=score(:,2)-score(:,3);
y=score2(:,2)-score2(:,3);
scatter(x,y,[],markerColor); hold on;
plot(xlim,[0,0],'k');
ylabel('diff IO R^2 (RC^+-RC^-)'); xlabel({'diff waveform R^2','(RC^+-RC^-)'});
set(gca,'fontname','calibri')
for i=1:10
    text(x(i)+0.015,y(i,1),sprintf('%d',i),'HorizontalAlignment','left','FontSize',8)
end
box off;

saveas(gcf,['figures' filesep 'fig_result_summary2.svg'])