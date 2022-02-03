filename = 'trj_1.csv';

data = readmatrix(filename);
time = data(:,1);
des_force = data(:,2:4);
xo = data(:,8:10);
force_vector = data(:,11:13);
xd = data(:,14:16);
pose = data(:,17:19);

%%
delta_t = 0.01;
dt = 2e-3;
N = delta_t/dt;
err = [];
for n = (N:max(size(time)))
    idx = 1+n-N;
    err(idx) =  des_force(n,3) -  force_vector(idx,3);
end

cuttime_start = 20;
cuttime_end = 40;
% cuttime_end = 110;

% cuttime_start = 0;
% cuttime_end = 200;
close all
fig = figure(1);
fig.Position = [100 100 600 400];
axes_fig = axes('Parent',fig);

grid on
hold on
plot(time - cuttime_start, des_force(:,3),'b','linewidth',3)
plot(time - cuttime_start, force_vector(:,3),'r--','linewidth',3)
% plot(time(1:max(size(time))-N +1), err,'k','linewidth',1)
plot(time - cuttime_start,des_force(:,3) - force_vector(:,3), 'k','linewidth',3)
legend('$||F_d||$', '$||F_e||$', '$||e_f||$','Interpreter','latex')
ylabel('||F||, H')
xlabel('t, c')
ylim([-20,70]);
xlim([0, cuttime_end - cuttime_start]);
set(axes_fig,'FontSize',20);
set(fig,'Units','Inches');
pos = get(fig,'Position');

print(fig,'./force_2.pdf','-dpdf','-r0')
print(fig,'./force_2.png','-dpng','-r300')

%%
err = sqrt(sum((xo(:,1:2)' - pose(:,1:2)').^2));

close all
fig = figure(1);
fig.Position = [100 100 600 400];
axes_fig = axes('Parent',fig);

grid on
hold on
plot(time - cuttime_start, err,'b','linewidth',3)
% plot(time, force_vector(:,3),'r--','linewidth',1)
% plot(time(1:max(size(time))-N +1), err,'k','linewidth',1)
% plot(time(1:max(size(time))-N + 1), err,'k','linewidth',1)
% legend('$||F_d||$', '$||F_e||$', '$||e_f||$','Interpreter','latex')
ylabel('||х||, м')
xlabel('t, c')
xlim([0, cuttime_end - cuttime_start]);
set(axes_fig,'FontSize',20);
set(fig,'Units','Inches');
pos = get(fig,'Position');

print(fig,'./pos_error_2.pdf','-dpdf','-r0')
print(fig,'./pos_error_2.png','-dpng','-r300')